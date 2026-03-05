import math
import torch
import chess
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament import Player, validate_player


class TransformerPlayer(Player):
    PIECE_VALUE = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }

    CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]

    def __init__(
        self,
        name: str = "Transformer-Qwen2.5-1.5B",
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        pretopk: int = 80,
        topk: int = 32,
        seed: int = 0,
        shield: bool = True,
        use_heuristics: bool = True,
        heuristic_blend: float = 0.15,
        use_chat_template: bool = True,
        load_in_8bit: bool = False,
        blunder_filter: bool = True,
        max_opponent_replies: int = 16,
        lm_batch_size: int = 2,
    ):
        super().__init__(name)

        self.model_name = model_name
        self.pretopk = int(pretopk)
        self.topk = int(topk)

        self.shield = bool(shield)
        self.use_heuristics = bool(use_heuristics)
        self.heuristic_blend = float(heuristic_blend)
        self.use_chat_template = bool(use_chat_template)

        self.blunder_filter = bool(blunder_filter)
        self.max_opponent_replies = int(max_opponent_replies)

        self.lm_batch_size = max(1, int(lm_batch_size))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            if load_in_8bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    load_in_8bit=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        try:
            self.model.config.use_cache = False
        except Exception:
            pass

        self.model.eval()

    def _make_prompt(self, board: chess.Board) -> str:
        fen = board.fen()
        stm = "White" if board.turn == chess.WHITE else "Black"
        user_msg = (
            f"Side to move: {stm}\n"
            f"FEN: {fen}\n"
            "Choose the best next move.\n"
            "Answer with UCI only (e.g., e2e4 or e7e8q)."
        )

        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_msg}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return (
            "You are a strong chess player.\n"
            f"Side to move: {stm}\n"
            f"FEN: {fen}\n"
            "Task: Choose the best next move. Answer with UCI only (e.g., e2e4 or e7e8q).\n"
            "Move: "
        )

    @torch.no_grad()
    def _score_uci_microbatch(self, prompt: str, ucis: list[str]) -> list[float]:
        if not ucis:
            return []

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = int(prompt_ids.shape[1])

        scores: list[float] = []
        bs = self.lm_batch_size

        for start in range(0, len(ucis), bs):
            chunk = ucis[start : start + bs]
            full_texts = [prompt + (" " + uci if not uci.startswith(" ") else uci) for uci in chunk]

            batch = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)

            out = self.model(input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits
            logprobs = torch.log_softmax(logits, dim=-1)

            B, T = input_ids.shape

            for b in range(B):
                seq_len = int(attn[b].sum().item())
                start_pred = prompt_len - 1
                end_pred = seq_len - 2
                if end_pred < start_pred:
                    scores.append(float("-inf"))
                    continue

                target = input_ids[b, prompt_len:seq_len]
                preds = logprobs[b, start_pred : end_pred + 1, :]
                token_lp = preds.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                scores.append(float(token_lp.sum().item()))

            del batch, input_ids, attn, out, logits, logprobs
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return scores

    def _find_mate_in_one(self, board: chess.Board) -> chess.Move | None:
        for mv in board.legal_moves:
            board.push(mv)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return mv
        return None

    def _opponent_has_mate_in_one(self, board_after_our_move: chess.Board) -> bool:
        for reply in board_after_our_move.legal_moves:
            board_after_our_move.push(reply)
            is_mate = board_after_our_move.is_checkmate()
            board_after_our_move.pop()
            if is_mate:
                return True
        return False

    def _hanging_major_piece_next_ply(self, board: chess.Board, our_move: chess.Move) -> bool:
        board.push(our_move)
        our_color = not board.turn

        majors = []
        for sq, piece in board.piece_map().items():
            if piece.color == our_color and piece.piece_type in (chess.QUEEN, chess.ROOK):
                majors.append(sq)

        if not majors:
            board.pop()
            return False

        for reply in board.legal_moves:
            if board.is_capture(reply) and reply.to_square in majors:
                board.pop()
                return True

        board.pop()
        return False

    def _apply_tactical_shield(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        if not moves:
            return moves

        safe_vs_mate = []
        for mv in moves:
            board.push(mv)
            opp_mates = self._opponent_has_mate_in_one(board)
            board.pop()
            if not opp_mates:
                safe_vs_mate.append(mv)
        if safe_vs_mate:
            moves = safe_vs_mate

        safe_major = []
        for mv in moves:
            if not self._hanging_major_piece_next_ply(board, mv):
                safe_major.append(mv)
        if safe_major:
            moves = safe_major

        return moves

    def _is_opening(self, board: chess.Board) -> bool:
        return board.fullmove_number <= 10

    def _starting_square(self, piece: chess.Piece, square: chess.Square) -> bool:
        if piece.piece_type == chess.KNIGHT:
            return square in ([chess.B1, chess.G1] if piece.color == chess.WHITE else [chess.B8, chess.G8])
        if piece.piece_type == chess.BISHOP:
            return square in ([chess.C1, chess.F1] if piece.color == chess.WHITE else [chess.C8, chess.F8])
        if piece.piece_type == chess.QUEEN:
            return square == (chess.D1 if piece.color == chess.WHITE else chess.D8)
        if piece.piece_type == chess.ROOK:
            return square in ([chess.A1, chess.H1] if piece.color == chess.WHITE else [chess.A8, chess.H8])
        if piece.piece_type == chess.KING:
            return square == (chess.E1 if piece.color == chess.WHITE else chess.E8)
        if piece.piece_type == chess.PAWN:
            rank = chess.square_rank(square)
            return rank == (1 if piece.color == chess.WHITE else 6)
        return False

    def _capture_gain(self, board: chess.Board, mv: chess.Move) -> float:
        if not board.is_capture(mv):
            return 0.0

        attacker = board.piece_at(mv.from_square)
        if attacker is None:
            return 0.0

        if board.is_en_passant(mv):
            victim_value = self.PIECE_VALUE[chess.PAWN]
        else:
            victim = board.piece_at(mv.to_square)
            victim_value = self.PIECE_VALUE[victim.piece_type] if victim else 0.0

        attacker_value = self.PIECE_VALUE[attacker.piece_type]
        return 0.4 + (victim_value - 0.2 * attacker_value)

    def _check_and_promo_bonus(self, board: chess.Board, mv: chess.Move) -> float:
        bonus = 0.0
        if mv.promotion is not None:
            bonus += 3.0 + self.PIECE_VALUE.get(mv.promotion, 0.0)
        board.push(mv)
        if board.is_check():
            bonus += 0.7
        board.pop()
        return bonus

    def _castling_bonus(self, board: chess.Board, mv: chess.Move) -> float:
        if board.is_castling(mv):
            return 1.2 if self._is_opening(board) else 0.6
        return 0.0

    def _development_bonus(self, board: chess.Board, mv: chess.Move) -> float:
        if not self._is_opening(board):
            return 0.0
        piece = board.piece_at(mv.from_square)
        if piece is None:
            return 0.0

        bonus = 0.0
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            from_rank = chess.square_rank(mv.from_square)
            if (piece.color == chess.WHITE and from_rank == 0) or (piece.color == chess.BLACK and from_rank == 7):
                bonus += 0.6

        if piece.piece_type == chess.QUEEN and self._starting_square(piece, mv.from_square):
            bonus -= 0.5

        return bonus

    def _repeat_piece_penalty(self, board: chess.Board, mv: chess.Move) -> float:
        if not self._is_opening(board):
            return 0.0
        piece = board.piece_at(mv.from_square)
        if piece is None:
            return 0.0
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.ROOK, chess.KING):
            if not self._starting_square(piece, mv.from_square):
                return -0.25
        return 0.0

    def _center_control_bonus(self, board: chess.Board, mv: chess.Move) -> float:
        bonus = 0.0
        board.push(mv)
        us = not board.turn

        for sq in self.CENTER_SQUARES:
            p = board.piece_at(sq)
            if p is not None and p.color == us:
                bonus += 0.35

        for sq in self.CENTER_SQUARES:
            bonus += 0.05 * len(board.attackers(us, sq))

        board.pop()
        return bonus

    def _moved_piece_hanging_penalty(self, board: chess.Board, mv: chess.Move) -> float:
        piece = board.piece_at(mv.from_square)
        if piece is None:
            return 0.0

        board.push(mv)
        moved_to = mv.to_square
        moved_piece = board.piece_at(moved_to)

        if mv.promotion is not None:
            moved_value = self.PIECE_VALUE.get(mv.promotion, 0.0)
        else:
            moved_value = self.PIECE_VALUE.get(moved_piece.piece_type, 0.0) if moved_piece else 0.0

        a = len(board.attackers(board.turn, moved_to))
        d = len(board.attackers(not board.turn, moved_to))
        board.pop()

        if a == 0:
            return 0.0
        if a > d:
            return -0.8 * moved_value * (a - d)
        return 0.0

    def _heuristic_score(self, board: chess.Board, mv: chess.Move) -> float:
        s = 0.0
        s += self._check_and_promo_bonus(board, mv)
        s += self._castling_bonus(board, mv)
        s += self._capture_gain(board, mv)
        s += self._development_bonus(board, mv)
        s += self._center_control_bonus(board, mv)
        s += self._repeat_piece_penalty(board, mv)
        s += self._moved_piece_hanging_penalty(board, mv)
        return s

    def _tactical_order_replies(self, board: chess.Board) -> list[chess.Move]:
        legal = list(board.legal_moves)
        checks, captures, promos, quiet = [], [], [], []

        for mv in legal:
            if mv.promotion is not None:
                promos.append(mv)
                continue
            if board.is_capture(mv):
                captures.append(mv)
                continue

            board.push(mv)
            is_check = board.is_check()
            board.pop()

            if is_check:
                checks.append(mv)
            else:
                quiet.append(mv)

        ordered = promos + checks + captures + quiet
        return ordered[: max(self.max_opponent_replies, 1)]

    def _allows_immediate_mate(self, board_after_our_move: chess.Board) -> bool:
        for reply in self._tactical_order_replies(board_after_our_move):
            board_after_our_move.push(reply)
            is_mate = board_after_our_move.is_checkmate()
            board_after_our_move.pop()
            if is_mate:
                return True
        return False

    def _opponent_can_capture_our_queen_next(self, board_after_our_move: chess.Board) -> bool:
        our_color = not board_after_our_move.turn
        queen_sqs = [
            sq
            for sq, p in board_after_our_move.piece_map().items()
            if p.color == our_color and p.piece_type == chess.QUEEN
        ]
        if not queen_sqs:
            return False

        for reply in self._tactical_order_replies(board_after_our_move):
            if board_after_our_move.is_capture(reply) and reply.to_square in queen_sqs:
                return True
        return False

    def _apply_2ply_blunder_filter(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        if not moves:
            return moves

        safe = []
        for mv in moves:
            board.push(mv)

            if self._allows_immediate_mate(board):
                board.pop()
                continue

            if self._opponent_can_capture_our_queen_next(board):
                board.pop()
                continue

            board.pop()
            safe.append(mv)

        return safe if safe else moves

    def get_move(self, fen: str):
        try:
            board = chess.Board(fen)
        except Exception:
            return None

        if self.shield:
            mate1 = self._find_mate_in_one(board)
            if mate1 is not None:
                return mate1.uci()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        if self.shield:
            filtered = self._apply_tactical_shield(board, legal_moves)
            if filtered:
                legal_moves = filtered

        if self.use_heuristics:
            scored = [(self._heuristic_score(board, mv), mv) for mv in legal_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            ordered = [mv for _, mv in scored]
        else:
            ordered = legal_moves

        ordered = ordered[: max(self.pretopk, 1)]
        if not ordered:
            return None

        if self.blunder_filter:
            ordered = self._apply_2ply_blunder_filter(board, ordered)

        candidates = ordered[: max(self.topk, 1)]
        if not candidates:
            return None

        prompt = self._make_prompt(board)
        ucis = [mv.uci() for mv in candidates]
        lm_scores = self._score_uci_microbatch(prompt, ucis)

        best_i = 0
        best_total = -1e30

        for i, mv in enumerate(candidates):
            lm = lm_scores[i]
            if self.use_heuristics and self.heuristic_blend != 0.0:
                h = self._heuristic_score(board, mv)
                total = lm + self.heuristic_blend * h
            else:
                total = lm

            if total > best_total:
                best_total = total
                best_i = i

        best_uci = ucis[best_i]

        try:
            if chess.Move.from_uci(best_uci) not in board.legal_moves:
                return None
        except Exception:
            return None

        return best_uciimport math
import torch
import chess
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament import Player


class RerankLMPlayer(Player):
    """
    Qwen2.5-1.5B-Instruct legal-move reranking player + tactical shield + heuristics
    + 2-ply blunder filter (avoid allowing immediate mate-in-1 and immediate queen loss)
    + MEMORY-SAFE LM scoring via micro-batching (prevents huge RAM spikes on large vocab models like Qwen)

    Required interface: get_move(self, fen: str) -> Optional[str]
    Returns: UCI string
    """

    PIECE_VALUE = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }

    CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]

    def __init__(
        self,
        name: str = "RerankLM-Qwen2.5-1.5B",
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        # Candidate sizes
        pretopk: int = 80,              # heuristic shortlist before 2-ply filter
        topk: int = 32,                 # candidates to LM-rerank
        # Randomness
        seed: int = 0,
        # Features
        shield: bool = True,
        use_heuristics: bool = True,
        heuristic_blend: float = 0.15,  # total = LM + blend * heuristic
        use_chat_template: bool = True,
        # Loading options
        load_in_8bit: bool = False,     # True if OOM (requires bitsandbytes)
        # 2-ply blunder filter options
        blunder_filter: bool = True,
        max_opponent_replies: int = 16, # limit opponent reply scan for speed
        # MEMORY CONTROL: micro-batch size for LM scoring
        lm_batch_size: int = 2,         # 1–4 recommended for Qwen in Colab
    ):
        super().__init__(name)

        self.model_name = model_name
        self.pretopk = int(pretopk)
        self.topk = int(topk)

        self.shield = bool(shield)
        self.use_heuristics = bool(use_heuristics)
        self.heuristic_blend = float(heuristic_blend)
        self.use_chat_template = bool(use_chat_template)

        self.blunder_filter = bool(blunder_filter)
        self.max_opponent_replies = int(max_opponent_replies)

        self.lm_batch_size = max(1, int(lm_batch_size))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            if load_in_8bit:
                # pip install bitsandbytes
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    load_in_8bit=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # IMPORTANT: disable KV cache during scoring (saves memory)
        try:
            self.model.config.use_cache = False
        except Exception:
            pass

        self.model.eval()

    # ---------------------------
    # Prompt
    # ---------------------------
    def _make_prompt(self, board: chess.Board) -> str:
        fen = board.fen()
        stm = "White" if board.turn == chess.WHITE else "Black"
        user_msg = (
            f"Side to move: {stm}\n"
            f"FEN: {fen}\n"
            "Choose the best next move.\n"
            "Answer with UCI only (e.g., e2e4 or e7e8q)."
        )

        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_msg}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return (
            "You are a strong chess player.\n"
            f"Side to move: {stm}\n"
            f"FEN: {fen}\n"
            "Task: Choose the best next move. Answer with UCI only (e.g., e2e4 or e7e8q).\n"
            "Move: "
        )

    # ---------------------------
    # MEMORY-SAFE micro-batched LM scoring
    # ---------------------------
    @torch.no_grad()
    def _score_uci_microbatch(self, prompt: str, ucis: list[str]) -> list[float]:
        """
        Score log P(uci | prompt) for each uci in ucis.
        Uses micro-batches to avoid enormous peak memory for large-vocab models (Qwen).
        """
        if not ucis:
            return []

        # Prompt token length once
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = int(prompt_ids.shape[1])

        scores: list[float] = []
        bs = self.lm_batch_size

        for start in range(0, len(ucis), bs):
            chunk = ucis[start : start + bs]
            full_texts = [prompt + (" " + uci if not uci.startswith(" ") else uci) for uci in chunk]

            batch = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = batch["input_ids"].to(self.device)          # [B,T]
            attn = batch["attention_mask"].to(self.device)          # [B,T]

            out = self.model(input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits                                     # [B,T,V]
            logprobs = torch.log_softmax(logits, dim=-1)             # [B,T,V]

            B, T = input_ids.shape

            for b in range(B):
                seq_len = int(attn[b].sum().item())
                # continuation tokens are positions [prompt_len .. seq_len-1]
                # predicted by logits positions [prompt_len-1 .. seq_len-2]
                start_pred = prompt_len - 1
                end_pred = seq_len - 2
                if end_pred < start_pred:
                    scores.append(float("-inf"))
                    continue

                target = input_ids[b, prompt_len:seq_len]                  # [C]
                preds = logprobs[b, start_pred : end_pred + 1, :]          # [C,V]
                token_lp = preds.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                scores.append(float(token_lp.sum().item()))

            # Encourage freeing memory promptly
            del batch, input_ids, attn, out, logits, logprobs
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return scores

    # ---------------------------
    # Tactical shield
    # ---------------------------
    def _find_mate_in_one(self, board: chess.Board) -> chess.Move | None:
        for mv in board.legal_moves:
            board.push(mv)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return mv
        return None

    def _opponent_has_mate_in_one(self, board_after_our_move: chess.Board) -> bool:
        for reply in board_after_our_move.legal_moves:
            board_after_our_move.push(reply)
            is_mate = board_after_our_move.is_checkmate()
            board_after_our_move.pop()
            if is_mate:
                return True
        return False

    def _hanging_major_piece_next_ply(self, board: chess.Board, our_move: chess.Move) -> bool:
        board.push(our_move)
        our_color = not board.turn

        majors = []
        for sq, piece in board.piece_map().items():
            if piece.color == our_color and piece.piece_type in (chess.QUEEN, chess.ROOK):
                majors.append(sq)

        if not majors:
            board.pop()
            return False

        for reply in board.legal_moves:
            if board.is_capture(reply) and reply.to_square in majors:
                board.pop()
                return True

        board.pop()
        return False

    def _apply_tactical_shield(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        if not moves:
            return moves

        safe_vs_mate = []
        for mv in moves:
            board.push(mv)
            opp_mates = self._opponent_has_mate_in_one(board)
            board.pop()
            if not opp_mates:
                safe_vs_mate.append(mv)
        if safe_vs_mate:
            moves = safe_vs_mate

        safe_major = []
        for mv in moves:
            if not self._hanging_major_piece_next_ply(board, mv):
                safe_major.append(mv)
        if safe_major:
            moves = safe_major

        return moves

    # ---------------------------
    # Heuristics
    # ---------------------------
    def _is_opening(self, board: chess.Board) -> bool:
        return board.fullmove_number <= 10

    def _starting_square(self, piece: chess.Piece, square: chess.Square) -> bool:
        if piece.piece_type == chess.KNIGHT:
            return square in ([chess.B1, chess.G1] if piece.color == chess.WHITE else [chess.B8, chess.G8])
        if piece.piece_type == chess.BISHOP:
            return square in ([chess.C1, chess.F1] if piece.color == chess.WHITE else [chess.C8, chess.F8])
        if piece.piece_type == chess.QUEEN:
            return square == (chess.D1 if piece.color == chess.WHITE else chess.D8)
        if piece.piece_type == chess.ROOK:
            return square in ([chess.A1, chess.H1] if piece.color == chess.WHITE else [chess.A8, chess.H8])
        if piece.piece_type == chess.KING:
            return square == (chess.E1 if piece.color == chess.WHITE else chess.E8)
        if piece.piece_type == chess.PAWN:
            rank = chess.square_rank(square)
            return rank == (1 if piece.color == chess.WHITE else 6)
        return False

    def _capture_gain(self, board: chess.Board, mv: chess.Move) -> float:
        if not board.is_capture(mv):
            return 0.0

        attacker = board.piece_at(mv.from_square)
        if attacker is None:
            return 0.0

        if board.is_en_passant(mv):
            victim_value = self.PIECE_VALUE[chess.PAWN]
        else:
            victim = board.piece_at(mv.to_square)
            victim_value = self.PIECE_VALUE[victim.piece_type] if victim else 0.0

        attacker_value = self.PIECE_VALUE[attacker.piece_type]
        return 0.4 + (victim_value - 0.2 * attacker_value)

    def _check_and_promo_bonus(self, board: chess.Board, mv: chess.Move) -> float:
        bonus = 0.0
        if mv.promotion is not None:
            bonus += 3.0 + self.PIECE_VALUE.get(mv.promotion, 0.0)
        board.push(mv)
        if board.is_check():
            bonus += 0.7
        board.pop()
        return bonus

    def _castling_bonus(self, board: chess.Board, mv: chess.Move) -> float:
        if board.is_castling(mv):
            return 1.2 if self._is_opening(board) else 0.6
        return 0.0

    def _development_bonus(self, board: chess.Board, mv: chess.Move) -> float:
        if not self._is_opening(board):
            return 0.0
        piece = board.piece_at(mv.from_square)
        if piece is None:
            return 0.0

        bonus = 0.0
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            from_rank = chess.square_rank(mv.from_square)
            if (piece.color == chess.WHITE and from_rank == 0) or (piece.color == chess.BLACK and from_rank == 7):
                bonus += 0.6

        if piece.piece_type == chess.QUEEN and self._starting_square(piece, mv.from_square):
            bonus -= 0.5

        return bonus

    def _repeat_piece_penalty(self, board: chess.Board, mv: chess.Move) -> float:
        if not self._is_opening(board):
            return 0.0
        piece = board.piece_at(mv.from_square)
        if piece is None:
            return 0.0
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.ROOK, chess.KING):
            if not self._starting_square(piece, mv.from_square):
                return -0.25
        return 0.0

    def _center_control_bonus(self, board: chess.Board, mv: chess.Move) -> float:
        bonus = 0.0
        board.push(mv)
        us = not board.turn

        for sq in self.CENTER_SQUARES:
            p = board.piece_at(sq)
            if p is not None and p.color == us:
                bonus += 0.35

        for sq in self.CENTER_SQUARES:
            bonus += 0.05 * len(board.attackers(us, sq))

        board.pop()
        return bonus

    def _moved_piece_hanging_penalty(self, board: chess.Board, mv: chess.Move) -> float:
        piece = board.piece_at(mv.from_square)
        if piece is None:
            return 0.0

        board.push(mv)
        moved_to = mv.to_square
        moved_piece = board.piece_at(moved_to)

        if mv.promotion is not None:
            moved_value = self.PIECE_VALUE.get(mv.promotion, 0.0)
        else:
            moved_value = self.PIECE_VALUE.get(moved_piece.piece_type, 0.0) if moved_piece else 0.0

        a = len(board.attackers(board.turn, moved_to))
        d = len(board.attackers(not board.turn, moved_to))
        board.pop()

        if a == 0:
            return 0.0
        if a > d:
            return -0.8 * moved_value * (a - d)
        return 0.0

    def _heuristic_score(self, board: chess.Board, mv: chess.Move) -> float:
        s = 0.0
        s += self._check_and_promo_bonus(board, mv)
        s += self._castling_bonus(board, mv)
        s += self._capture_gain(board, mv)
        s += self._development_bonus(board, mv)
        s += self._center_control_bonus(board, mv)
        s += self._repeat_piece_penalty(board, mv)
        s += self._moved_piece_hanging_penalty(board, mv)
        return s

    # ---------------------------
    # 2-ply blunder filter
    # ---------------------------
    def _tactical_order_replies(self, board: chess.Board) -> list[chess.Move]:
        legal = list(board.legal_moves)
        checks, captures, promos, quiet = [], [], [], []

        for mv in legal:
            if mv.promotion is not None:
                promos.append(mv)
                continue
            if board.is_capture(mv):
                captures.append(mv)
                continue

            board.push(mv)
            is_check = board.is_check()
            board.pop()

            if is_check:
                checks.append(mv)
            else:
                quiet.append(mv)

        ordered = promos + checks + captures + quiet
        return ordered[: max(self.max_opponent_replies, 1)]

    def _allows_immediate_mate(self, board_after_our_move: chess.Board) -> bool:
        for reply in self._tactical_order_replies(board_after_our_move):
            board_after_our_move.push(reply)
            is_mate = board_after_our_move.is_checkmate()
            board_after_our_move.pop()
            if is_mate:
                return True
        return False

    def _opponent_can_capture_our_queen_next(self, board_after_our_move: chess.Board) -> bool:
        our_color = not board_after_our_move.turn
        queen_sqs = [sq for sq, p in board_after_our_move.piece_map().items()
                     if p.color == our_color and p.piece_type == chess.QUEEN]
        if not queen_sqs:
            return False

        for reply in self._tactical_order_replies(board_after_our_move):
            if board_after_our_move.is_capture(reply) and reply.to_square in queen_sqs:
                return True
        return False

    def _apply_2ply_blunder_filter(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        if not moves:
            return moves

        safe = []
        for mv in moves:
            board.push(mv)

            # opponent to move
            if self._allows_immediate_mate(board):
                board.pop()
                continue

            if self._opponent_can_capture_our_queen_next(board):
                board.pop()
                continue

            board.pop()
            safe.append(mv)

        return safe if safe else moves

    # ---------------------------
    # Required interface
    # ---------------------------
    def get_move(self, fen: str):
        try:
            board = chess.Board(fen)
        except Exception:
            return None

        # 1) mate in 1
        if self.shield:
            mate1 = self._find_mate_in_one(board)
            if mate1 is not None:
                return mate1.uci()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # 2) tactical shield
        if self.shield:
            filtered = self._apply_tactical_shield(board, legal_moves)
            if filtered:
                legal_moves = filtered

        # 3) heuristic ordering
        if self.use_heuristics:
            scored = [(self._heuristic_score(board, mv), mv) for mv in legal_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            ordered = [mv for _, mv in scored]
        else:
            ordered = legal_moves

        ordered = ordered[: max(self.pretopk, 1)]
        if not ordered:
            return None

        # 4) 2-ply blunder filter (on shortlist)
        if self.blunder_filter:
            ordered = self._apply_2ply_blunder_filter(board, ordered)

        # 5) LM rerank
        candidates = ordered[: max(self.topk, 1)]
        if not candidates:
            return None

        prompt = self._make_prompt(board)
        ucis = [mv.uci() for mv in candidates]
        lm_scores = self._score_uci_microbatch(prompt, ucis)

        best_i = 0
        best_total = -1e30

        for i, mv in enumerate(candidates):
            lm = lm_scores[i]
            if self.use_heuristics and self.heuristic_blend != 0.0:
                h = self._heuristic_score(board, mv)
                total = lm + self.heuristic_blend * h
            else:
                total = lm

            if total > best_total:
                best_total = total
                best_i = i

        best_uci = ucis[best_i]

        # final safety
        try:
            if chess.Move.from_uci(best_uci) not in board.legal_moves:
                return None
        except Exception:
            return None

        return best_uci
