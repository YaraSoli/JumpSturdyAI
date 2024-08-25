import copy
import random
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

transposition_table = {}


class Pos:
    """
       Initializes a position on the board.

       Parameters:
       - row (int): The row index (0-7).
       - col (int): The column index (0-7).
       """
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def to_chess_notation(self):
        """
        Converts the position to chess notation (e.g., A1, B2).

        Returns:
        - str: The chess notation of the position.
        """
        # Convert row index to chess notation (A-H)
        chess_col = chr(ord('A') + self.col)
        # Convert column index to chess notation (1-8)
        chess_row = str(self.row + 1)
        # Combine row and column notation
        return chess_col + chess_row


class Bitboards:
    def __init__(self):
        """
        Initializes the bitboards for the game. Each bitboard represents a different type of piece or state.
        """
        self.r = 0  # Bitboard for single red pieces
        self.b = 0  # Bitboard for single blue pieces
        self.rb = 0  # Bitboard for red-blue stack pieces
        self.br = 0  # Bitboard for blue-red stack pieces
        self.bb = 0  # Bitboard for double blue pieces
        self.rr = 0  # Bitboard for double red pieces
        self.all_pieces = 0  # Bitboard for all pieces

    def set_piece(self, piece_type, row, col):
        """
        Sets a piece on the board.

        Parameters:
        - piece_type (str): The type of piece ('r', 'b', 'rb', 'br', 'bb', 'rr').
        - row (int): The row index (0-7).
        - col (int): The column index (0-7).
        """
        # Calculate the bit position based on row and column
        bit_position = row * 8 + col
        if piece_type == 'r':
            self.r |= 1 << bit_position  # Set the bit in the 'r' bitboard - it adds 1 in the r board where the position is
        elif piece_type == 'b':
            self.b |= 1 << bit_position  # Set the bit in the 'b' bitboard
        elif piece_type == 'rb':
            self.rb |= 1 << bit_position  # Set the bit in the 'rb' bitboard
        elif piece_type == 'br':
            self.br |= 1 << bit_position  # Set the bit in the 'br' bitboard
        elif piece_type == 'bb':
            self.bb |= 1 << bit_position  # Set the bit in the 'bb' bitboard
        elif piece_type == 'rr':
            self.rr |= 1 << bit_position  # Set the bit in the 'rr' bitboard
        # Update the all_pieces bitboard
        self.all_pieces |= 1 << bit_position

    def get_piece(self, row, col):
        """
         Retrieves the type of piece at a given position.

         Parameters:
         - row (int): The row index (0-7).
         - col (int): The column index (0-7).

         Returns:
         - str: The type of piece at the position ('r', 'b', 'rb', 'br', 'bb', 'rr'), or None if empty.
         """
        bit_position = row * 8 + col # Calculate bit position based on row and column

        if self.rr & (1 << bit_position):
            return 'rr'
        elif self.br & (1 << bit_position):
            return 'br'
        elif self.bb & (1 << bit_position):
            return 'bb'
        elif self.rb & (1 << bit_position):
            return 'rb'
        elif self.r & (1 << bit_position):
            return 'r'
        elif self.b & (1 << bit_position):
            return 'b'

        return None  # No piece found at this position

    def remove_piece(self, row, col):
        """
         Removes a piece from the board at the specified position.

         Parameters:
         - row (int): The row index (0-7).
         - col (int): The column index (0-7).
         """
        bit_position = row * 8 + col
        self.r &= ~(1 << bit_position)
        self.b &= ~(1 << bit_position)
        self.rb &= ~(1 << bit_position)
        self.br &= ~(1 << bit_position)
        self.bb &= ~(1 << bit_position)
        self.rr &= ~(1 << bit_position)
        self.all_pieces &= ~(1 << bit_position)

    def combine_boards(self):
        """
        Combines all individual bitboards into a single 2D board.

        Returns:
        - list: A 2D list representing the combined board.
        """
        combined = [["." for _ in range(8)] for _ in range(8)]
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece:
                    combined[row][col] = piece
        # Set corners to '0'
        combined[0][0] = '0'
        combined[0][7] = '0'
        combined[7][0] = '0'
        combined[7][7] = '0'
        return combined

    def print_combined_board(self):
        """
         Prints the combined board with chess notation.
         """
        combined_board = self.combine_boards()
        print("Combined Board:")
        for row_idx, row in enumerate(reversed(combined_board)):  # Print rows from 7 to 0
            print("\x1b[95m{}\x1b[0m".format(8 - row_idx), end=" ")  # Pink color for row numbers
            for piece in row:
                if piece == 'r':
                    print("\x1b[31mr\x1b[0m", end=" ")  # Red 'r'
                elif piece == 'b':
                    print("\x1b[34mb\x1b[0m", end=" ")  # Blue 'b'
                elif piece == 'rb':
                    print("\x1b[34mrb\x1b[0m", end=" ")  # Blue 'rb'
                elif piece == 'br':
                    print("\x1b[31mbr\x1b[0m", end=" ")  # Red 'br'
                elif piece == 'bb':
                    print("\x1b[34mbb\x1b[0m", end=" ")  # Blue 'bb'
                elif piece == 'rr':
                    print("\x1b[31mrr\x1b[0m", end=" ")  # Red 'rr'
                else:
                    print(piece, end=" ")
            print()
        print("\x1b[95m{}\x1b[0m".format("  A B C D E F G H"))  # Pink color for column headers

    def get_all_pieces(self, player):
        """
        Gets all positions of the pieces for a given player.

        Parameters:
        - player (str): The player ('r' for red, 'b' for blue).

        Returns:
        - list: A list of tuples representing the positions of the player's pieces.
        """

        # Select the correct bitboard for the current player ('r' for red, 'b' for blue)
        bitboard = self.r if player == 'r' else self.b

        # Initialize an empty list to hold the positions of the pieces
        positions = []

        # Iterate over each bit position in the bitboard (0 to 63 for an 8x8 board)
        for i in range(64):
            # Check if the bit at position i is set (indicating a piece is present)
            if bitboard & (1 << i):
                # Calculate the row and column from the bit index and add to the positions list
                positions.append((i // 8, i % 8))

        # Return the list of positions where pieces are present for the given player
        return positions


def reformulate(fen):
    # Adds padding to the first and last rows in the FEN string.
    rows = fen.split("/")
    rows[0] = "1" + rows[0] + "1"
    rows[7] = "1" + rows[7] + "1"
    new_fen = "/".join(rows)
    return new_fen


def reverse_fen(fen):
    # Reverses the board representation in a FEN string and preserves the player's turn.

    # Split the FEN string into parts
    parts = fen.split(' ')
    position = parts[0]
    color = parts[1] if len(parts) > 1 else None

    # Split the position into rows and reverse the order
    rows = position.split('/')
    reversed_rows = rows[::-1]

    # Join the reversed rows back into a position string
    reversed_position = '/'.join(reversed_rows)

    # Reconstruct the FEN string
    reversed_fen = reversed_position
    if color:
        reversed_fen += f' {color}'

    return reversed_fen


def parse_fen(fen):
    # Converts a FEN string into a Bitboards object.

    bitboards = Bitboards()
    rows = fen.split('/')
    rows.reverse()  # Reverse the rows to parse from bottom to top

    for row_num, row in enumerate(rows):
        col = 0
        i = 7 - row_num  # Map the FEN row to the board row, starting from the top (7th rank) 0

        j = 0
        while j < len(row):
            char = row[j]
            if col >= 8:
                break  # Prevent going out of column bounds

            if char == 'r':
                if j < len(row) - 1 and row[j + 1] == '0':
                    bitboards.set_piece('r', i, col)
                    j += 1  # Skip the next character
                elif j < len(row) - 1 and row[j + 1] == 'b':
                    bitboards.set_piece('rb', i, col)
                    j += 1  # Skip the next character
                elif j < len(row) - 1 and row[j + 1] == 'r':
                    bitboards.set_piece('rr', i, col)
                    j += 1  # Skip the next character
                else:
                    bitboards.set_piece('r', i, col)
            elif char == 'b':
                if j < len(row) - 1 and row[j + 1] == '0':
                    bitboards.set_piece('b', i, col)
                    j += 1  # Skip the next character
                elif j < len(row) - 1 and row[j + 1] == 'r':
                    bitboards.set_piece('br', i, col)
                    j += 1  # Skip the next character
                elif j < len(row) - 1 and row[j + 1] == 'b':
                    bitboards.set_piece('bb', i, col)
                    j += 1  # Skip the next character
                else:
                    bitboards.set_piece('b', i, col)
            elif char.isdigit():
                col += int(char)  # Skip the corresponding number of columns
                j += 1
                continue  # Skip the increment of col below

            col += 1
            j += 1

    return bitboards


def check_game_end(bitboards):
    # Checks whether the game has ended based on the positions of the pieces on the board.

    # Masks to isolate the top row (row 0) and bottom row (row 7)
    top_row_mask = 0xFF00000000000000
    bottom_row_mask = 0x00000000000000FF

    # Check if any red pieces are in the top row (row 0)
    if (bitboards.b & top_row_mask) or (bitboards.rb & top_row_mask) or (bitboards.bb & top_row_mask):
        return "Game Over: Blue wins :D"

    # Check if any blue pieces are in the bottom row (row 7)
    if (bitboards.r & bottom_row_mask) or (bitboards.br & bottom_row_mask) or (bitboards.rr & bottom_row_mask):
        return "Game Over: Red wins :D"

    return None  # Game is not over yet


def calculate_possible_moves_for_stack(bitboards, row, col, player):
    # Calculates possible moves for stacked pieces based on their player type.

    forbidden_positions = [(0, 0), (7, 7), (0, 7), (7, 0)]

    possible_moves = []

    # Directions for stack piece movements
    if player == 'b':
        move_directions = [(2, 1), (2, -1), (1, -2), (1, 2)]
    else:
        move_directions = [(-2, -1), (-1, -2), (-1, 2), (-2, 1)]

    # Iterate over possible move directions
    for dr, dc in move_directions:
        r, c = row, col

        # Calculate the new position
        r_new = r + dr
        c_new = c + dc

        # Check if the move is within board boundaries
        if 0 <= r_new < 8 and 0 <= c_new < 8 and (r_new, c_new) not in forbidden_positions:
            bit_position = r_new * 8 + c_new

            # Check if the position is empty or contains exactly one piece
            if ((bitboards.r & (1 << bit_position)) ^ (bitboards.b & (1 << bit_position))) or \
                    ((bitboards.br & (1 << bit_position)) and player == 'b') or \
                    ((bitboards.rb & (1 << bit_position)) and player == 'r') or \
                    ((bitboards.bb & (1 << bit_position)) and player == 'r') or \
                    ((bitboards.rr & (1 << bit_position)) and player == 'b') or \
                    (not (bitboards.all_pieces & (1 << bit_position))):
                possible_moves.append((r_new, c_new))

    return possible_moves


# Function to calculate possible moves
def calculate_possible_moves(bitboard, row, col, player):
    forbidden_positions = [(0, 0), (7, 7), (0, 7), (7, 0)]
    possible_moves = []
    piece = bitboard.get_piece(row, col)

    if piece in ['rr', 'br', 'bb', 'rb']:

        return calculate_possible_moves_for_stack(bitboard, row, col, player)

    if piece == 'b':
        move_directions = [(0, 1), (0, -1), (1, 0)]
        capture_directions = [(1, -1), (1, 1)]
    elif piece == 'r':
        move_directions = [(-1, 0), (0, 1), (0, -1)]
        capture_directions = [(-1, 1), (-1, -1)]
    else:
        return []

    # Iterate over possible move directions
    for dr, dc in move_directions:
        r, c = row + dr, col + dc
        bit_position = r * 8 + c
        # Check if the move is within bounds and the square is empty
        if 0 <= r < 8 and 0 <= c < 8 and (r, c) not in forbidden_positions:
            if not (bitboard.all_pieces & (1 << bit_position)):
                possible_moves.append((r, c))  # if there is no piece then move
            elif piece == 'r' and (bitboard.r & (1 << bit_position)):  # occupied by a single red
                possible_moves.append((r, c))
            elif piece == 'b' and (bitboard.b & (1 << bit_position)):
                possible_moves.append((r, c))

    # Iterate over possible capture directions
    for dr, dc in capture_directions:
        r, c = row + dr, col + dc
        bit_position = r * 8 + c
        # Check if the move is within bounds and there's an opponent's piece
        if (0 <= r < 8 and 0 <= c < 8 and
                bitboard.all_pieces & (1 << (r * 8 + c)) and
                (r, c) not in forbidden_positions):
            if player == 'r' and (
                    (bitboard.b & (1 << bit_position)) != 0 or
                    (bitboard.rb & (1 << bit_position)) != 0 or
                    (bitboard.bb & (1 << bit_position)) != 0):
                possible_moves.append((r, c))
            elif player == 'b' and (
                    (bitboard.r & (1 << bit_position)) != 0 or
                    (bitboard.br & (1 << bit_position)) != 0 or
                    (bitboard.rr & (1 << bit_position)) != 0):
                possible_moves.append((r, c))

    return possible_moves


def calculate_all_possible_moves(bitboards, player):
    """
        Calculates all possible moves for the given player.

        Parameters:
        - bitboards: The current state of the board.
        - player: The player ('r' for red, 'b' for blue) whose moves are to be calculated.

        Returns:
        - all_possible_moves: A dictionary with positions as keys and lists of possible moves as values.
        """
    all_possible_moves = {}

    for row in range(8):
        for col in range(8):
            piece = bitboards.get_piece(row, col)
            if piece and piece[-1] == player:
                possible_moves = calculate_possible_moves(bitboards, row, col, player)
                all_possible_moves[(row, col)] = possible_moves

    return all_possible_moves


def do_move(start_pos, end_pos, player, bitboards):
    """
       Executes a move on the board and returns the updated bitboards.

       Parameters:
       - start_pos: The starting position of the move (Pos object).
       - end_pos: The ending position of the move (Pos object).
       - player: The player making the move ('r' or 'b').
       - bitboards: The current state of the board.

       Returns:
       - updated_bitboards: The board state after the move.
       """
    # Create a deep copy of the bitboards
    updated_bitboards = copy.deepcopy(bitboards)  # max recursion reach

    # Get the piece type at the start position
    start_piece_type = updated_bitboards.get_piece(start_pos.row, start_pos.col)
    if not start_piece_type:
        print(f"No piece at start position ({start_pos.row}, {start_pos.col})")
    else:
        # Remove the piece from the start position
        updated_bitboards.remove_piece(start_pos.row, start_pos.col)

        # Determine the correct piece type to place at the start position
        if start_piece_type == 'rr' and player == 'r':
            updated_bitboards.set_piece('r', start_pos.row, start_pos.col)
        elif start_piece_type == 'br' and player == 'r':
            updated_bitboards.set_piece('b', start_pos.row, start_pos.col)
        elif start_piece_type == 'rb' and player == 'b':
            updated_bitboards.set_piece('r', start_pos.row, start_pos.col)
        elif start_piece_type == 'bb' and player == 'b':
            updated_bitboards.set_piece('b', start_pos.row, start_pos.col)

    # Handle the end position based on the type of piece currently there
    end_piece_type = updated_bitboards.get_piece(end_pos.row, end_pos.col)

    if end_piece_type:
        # Handle capturing and replacing pieces based on the player's move
        if end_piece_type == 'r' and player == 'b':
            updated_bitboards.remove_piece(end_pos.row, end_pos.col)
            updated_bitboards.set_piece('b', end_pos.row, end_pos.col)
        elif end_piece_type == 'r' and player == 'r':
            updated_bitboards.remove_piece(end_pos.row, end_pos.col)
            updated_bitboards.set_piece('rr', end_pos.row, end_pos.col)
        elif end_piece_type == 'b' and player == 'r':
            updated_bitboards.remove_piece(end_pos.row, end_pos.col)
            updated_bitboards.set_piece('r', end_pos.row, end_pos.col)
        elif end_piece_type == 'b' and player == 'b':
            updated_bitboards.remove_piece(end_pos.row, end_pos.col)
            updated_bitboards.set_piece('bb', end_pos.row, end_pos.col)
        elif end_piece_type == 'rb' and player == 'r':
            updated_bitboards.remove_piece(end_pos.row, end_pos.col)
            updated_bitboards.set_piece('rr', end_pos.row, end_pos.col)
        elif end_piece_type == 'bb' and player == 'r':
            updated_bitboards.remove_piece(end_pos.row, end_pos.col)
            updated_bitboards.set_piece('br', end_pos.row, end_pos.col)
        elif end_piece_type == 'rr' and player == 'b':
            updated_bitboards.remove_piece(end_pos.row, end_pos.col)
            updated_bitboards.set_piece('rb', end_pos.row, end_pos.col)
        elif end_piece_type == 'br' and player == 'b':
            updated_bitboards.remove_piece(end_pos.row, end_pos.col)
            updated_bitboards.set_piece('bb', end_pos.row, end_pos.col)

    else:
        # If there's no piece at the end position, place the player's piece there
        if player == 'r':
            updated_bitboards.set_piece('r', end_pos.row, end_pos.col)
        else:
            updated_bitboards.set_piece('b', end_pos.row, end_pos.col)
    return updated_bitboards


def calculate_score(bitboards, player, move=None):
    """
    Calculates the score for a given player based on the current board state.

    Parameters:
    - bitboards: The current state of the board.
    - player: The player whose score is to be calculated ('r' or 'b').
    - move: Optional move tuple (start_pos, end_pos) to evaluate the move impact.

    Returns:
    - score: An integer representing the score.
    """
    score = 0
    piece_values = {'r': 1, 'b': 1, 'rr': 2, 'bb': 2, 'br': 2, 'rb': 2}  # Base values for pieces
    capture_bonus = 5  # Bonus score for capturing an opponent's piece

    # Set opponent and direction vectors based on the player
    if player == 'r':
        opponent = 'b'
        diagonal_directions = [(-1, -1), (-1, 1)]
        horse_directions = [(-2, 1), (-2, -1), (-1, -2), (-1, 2)]  # y, x
    else:
        opponent = 'r'
        diagonal_directions = [(1, 1), (1, -1)]
        horse_directions = [(1, -2), (1, 2), (2, -1), (2, 1)]

    if move:
        start_pos, end_pos = move
        end_piece = bitboards.get_piece(end_pos[0], end_pos[1])
        if end_piece and end_piece[0] == opponent:
            score += capture_bonus # Add bonus for capturing an opponent's piece

    pieces = bitboards.get_all_pieces(player)
    for row, col in pieces:
        piece = bitboards.get_piece(row, col)

     # Determine the score based on proximity to the finish line
        if player == 'b':  # Red player aims for the top row
            proximity_score = (row - 0)  # Score increases as red moves upwards
        elif player == 'r':  # Blue player aims for the bottom row
            proximity_score = (7 - row)  # Score increases as blue moves downwards

        # Calculate the base score using piece values and proximity
        base_score = piece_values.get(piece[0], 0) * proximity_score
        score += base_score

        # Check for threats from diagonals
        for dr, dc in diagonal_directions:
            threat_row, threat_col = row + dr, col + dc
            if 0 <= threat_row < 8 and 0 <= threat_col < 8:
                threat_piece = bitboards.get_piece(threat_row, threat_col)
                if threat_piece and threat_piece[0] == opponent:
                    score -= piece_values.get(piece[0], 0)  # Subtract score if threatened by opponent
                    break
                elif threat_piece and threat_piece[0] == player:
                    score += 1  # Add score if the player's piece is in a diagonal position

        # Check for threats from horse directions
        for dr, dc in horse_directions:
            threat_row, threat_col = row + dr, col + dc
            if 0 <= threat_row < 8 and 0 <= threat_col < 8:
                threat_piece = bitboards.get_piece(threat_row, threat_col)
                if threat_piece and threat_piece[0] == opponent:

                    score -= piece_values.get(piece[0], 0)  # Subtract score if threatened by two opponents
                    break  # Only need to subtract once per piece
                elif threat_piece and threat_piece[0] == player:
                    score += 2  # Add score if the player's piece is in a horse position

    return score


def alpha_beta(bitboards, alpha, beta, depth, player, start_time, time_limit, move=None):
    """
    Implements the Alpha-Beta pruning algorithm for minimax decision-making.

    Parameters:
    - bitboards: The current state of the board.
    - alpha: The best score that the maximizer can guarantee.
    - beta: The best score that the minimizer can guarantee.
    - depth: The maximum depth of the search tree.
    - player: The current player ('r' for red, 'b' for blue).
    - start_time: The time when the search started.
    - time_limit: The maximum time allowed for the search.
    - move: The move being considered (optional).

    Returns:
    - A tuple of (score, best_move) where:
        - score: The evaluated score of the board.
        - best_move: The best move found at the current depth.
    """
    if time.time() - start_time > time_limit:
        # If time limit exceeded, return the current score
        if player == 'r':
            return calculate_score(bitboards, 'b', move), None
        else:
            return calculate_score(bitboards, 'r', move), None

    game_end_status = check_game_end(bitboards)
    if game_end_status:
        # Check if the game is over and return the appropriate extreme score
        if game_end_status == "Game Over: Blue wins :D":
            if player == 'b':
                return float('inf'), None
            else:
                return float('-inf'), None
        if game_end_status == "Game Over: Red wins :D":
            if player == 'r':
                return float('inf'), None
            else:
                return float('-inf'), None

    if depth == 0:
        # If maximum depth reached, return the evaluated score
        if player == 'r':
            return calculate_score(bitboards, 'b', move), None
        else:
            return calculate_score(bitboards, 'r', move), None

    best_move = None
    all_possible_moves = calculate_all_possible_moves(bitboards, player)
    if player == 'r':
        max_eval = float('-inf')  # Initialize maximum evaluation for maximizer
        for start_pos, moves in all_possible_moves.items():
            for end_pos in moves:
                updated_bitboards = do_move(Pos(*start_pos), Pos(*end_pos), player, bitboards)
                move = (start_pos, end_pos)
                eval, _ = alpha_beta(updated_bitboards, alpha, beta, depth - 1, 'b', start_time, time_limit, move)
                if eval > max_eval:
                    max_eval = eval
                    best_move = (start_pos, end_pos)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break # Beta cutoff
        return max_eval, best_move
    else:
        min_eval = float('inf') # Initialize minimum evaluation for minimizer
        for start_pos, moves in all_possible_moves.items():
            for end_pos in moves:
                updated_bitboards = do_move(Pos(*start_pos), Pos(*end_pos), player, bitboards)
                eval, _ = alpha_beta(updated_bitboards, alpha, beta, depth - 1, 'r', start_time, time_limit)
                if eval < min_eval:
                    min_eval = eval
                    best_move = (start_pos, end_pos)
                beta = min(beta, eval)
                if beta <= alpha:
                    break # Alpha cutoff
    return min_eval, best_move


def game_stage(bitboards):
    """
    Determines the current stage of the game based on the number of pieces on the board.

    Parameters:
    - bitboards: The current state of the board.

    Returns:
    - A string representing the stage of the game: 'early', 'mid', or 'late'.
    """
    # Count pieces on the board
    num_red_pieces = bin(bitboards.r).count('1') + bin(bitboards.rr).count('1')
    num_blue_pieces = bin(bitboards.b).count('1') + bin(bitboards.bb).count('1')

    total_pieces = num_red_pieces + num_blue_pieces

    # Thresholds for determining the stage based on the number of pieces
    if total_pieces > 16:
        return 'early'
    elif 8 < total_pieces <= 16:
        return 'mid'
    else:
        return 'late'


def dynamic_time_allocation(stage, base_time):
    """
    Allocates time dynamically based on the game stage.

    Parameters:
    - stage: The current stage of the game ('early', 'mid', 'late').
    - base_time: The base time allocated for the move.

    Returns:
    - The adjusted time allocation.
    """
    if stage == 'early':
        return base_time * 0.5
    elif stage == 'mid':
        return base_time * 1.5
    else:
        return base_time * 0.75


def iterative_deepening(bitboards, player, total_time):
    """
    Implements the iterative deepening search strategy to find the best move.

    Parameters:
    - bitboards: The current state of the board.
    - player: The current player ('r' for red, 'b' for blue).
    - total_time: The total time allocated for the search.

    Returns:
    - A tuple of (best_score, best_move) where:
        - best_score: The best score found.
        - best_move: The best move found.
    """
    start_time = time.time()
    depth = 1
    best_move = None
    best_score = float('-inf') if player == 'r' else float('inf')

    stage = game_stage(bitboards)
    time_per_move = dynamic_time_allocation(stage, total_time)

    while time.time() - start_time < time_per_move:
        time_left = time_per_move - (time.time() - start_time)
        if time_left <= 0:
            break
        score, move = alpha_beta(bitboards, float('-inf'), float('inf'), depth, player, start_time, time_left)
        if time.time() - start_time < total_time and move:
            start_pos, end_pos = move
            #print("Depth = ", depth)
            #print(f"move: {Pos(*start_pos).to_chess_notation()} to {Pos(*end_pos).to_chess_notation()}", "score: ",
            #      score)

            if (player == 'r' and score > best_score) or (player == 'b' and score < best_score):
                best_score = score
                best_move = move

        depth += 1 # Increase depth for next iteration

    return best_score, best_move


def simulate_game(fen_player, total_time=120):
    """
    Simulates a full game from a given FEN string using iterative deepening search.

    Parameters:
    - fen_player: The FEN string representing the board state and the current player.
    - total_time: The total time allocated for the game simulation.

    Returns:
    - None
    """
    player = fen_player[-1] # Extract player from FEN string
    fen = fen_player[:-2] # Extract FEN board state
    bitboards = parse_fen(reformulate(fen))
    start_time = time.time()

    while time.time() - start_time < total_time:
        if check_game_end(bitboards):
            break # End simulation if game is over

        move_time = min(total_time - (time.time() - start_time), 1.0)  # Allocate 1 second for each move
        best_score, best_move = iterative_deepening(bitboards, player, move_time)

        if not best_move:
            print("No valid move found", player, "Lost")
            break

        start_pos, end_pos = best_move
        bitboards = do_move(Pos(*start_pos), Pos(*end_pos), player, bitboards)
        print(
            f"{player.upper()} moved from {Pos(*start_pos).to_chess_notation()} to {Pos(*end_pos).to_chess_notation()}")
        bitboards.print_combined_board()

        player = 'b' if player == 'r' else 'r' # Switch player

    print("Game Over")
    game_end_status = check_game_end(bitboards)
    if game_end_status:
        print(game_end_status)
    else:
        print("Time limit reached")


def fitness_function_alpha_beta(bitboards, player, move, move_time):
    """
    Evaluates the fitness of a move using Alpha-Beta pruning within an evolutionary algorithm.

    Parameters:
    - bitboards: The current state of the board.
    - player: The current player ('r' for red, 'b' for blue).
    - move: The move being evaluated.
    - move_time: The time allocated for the move evaluation.

    Returns:
    - The fitness score of the move.
    """
    start_pos, end_pos = move
    simulated_bitboards = do_move(Pos(*start_pos), Pos(*end_pos), player, bitboards)
    board_hash = hash(simulated_bitboards)

    if board_hash in transposition_table:
        return transposition_table[board_hash] # Return cached result if available

    opponent = 'b' if player == 'r' else 'r'
    score, _ = iterative_deepening(simulated_bitboards, opponent, move_time)

    transposition_table[board_hash] = score # Cache result for future use
    return score


def evaluate_population(bitboards, player, population, move_time, total_moves):
    """
     Evaluates the fitness of a population of moves using parallel processing.

     Parameters:
     - bitboards: The current state of the board.
     - player: The current player ('r' for red, 'b' for blue).
     - population: The list of moves to be evaluated.
     - move_time: The time allocated for each move evaluation.
     - total_moves: The total number of moves to evaluate.

     Returns:
     - A list of fitness scores for the population.
     """
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(fitness_function_alpha_beta, bitboards, player, move, move_time, total_moves) for
                   move in population]
        fitnesses = [future.result() for future in futures]
    return fitnesses


# Initialpopulation erzeugen
def generate_initial_population(bitboards, player, pop_size):
    """
    Generates the initial population of moves for the evolutionary algorithm.

    Parameters:
    - bitboards: The current state of the board.
    - player: The current player ('r' for red, 'b' for blue).
    - pop_size: The desired size of the population.

    Returns:
    - A list of randomly selected moves.
    """
    population = []
    all_possible_moves = calculate_all_possible_moves(bitboards, player)
    move_list = [(start_pos, end_pos) for start_pos in all_possible_moves for end_pos in all_possible_moves[start_pos]]
    for _ in range(pop_size):
        population.append(random.choice(move_list))  # we choose random 3 moves
    return population


def select_parents(population, fitnesses, num_parents):
    """
    Selects the best individuals from the population to serve as parents for the next generation.

    Parameters:
    - population: The list of moves.
    - fitnesses: The list of fitness scores corresponding to the population.
    - num_parents: The number of parents to select.

    Returns:
    - A list of selected parent moves.
    """
    parents = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [parent[0] for parent in parents[:num_parents]]


def crossover(parents, num_offspring):
    """
    Performs crossover to generate offspring from selected parents.

    Parameters:
    - parents: The list of selected parent moves.
    - num_offspring: The number of offspring to generate.

    Returns:
    - A list of offspring moves.
    """
    offspring = []
    for _ in range(num_offspring):
        parent1, parent2 = random.sample(parents, 2)
        start_pos = parent1[0]
        end_pos = parent1[1]
        offspring.append((start_pos, end_pos))
    return offspring


def mutate(bitboards, offspring, mutation_rate, player):
    """
    Mutates the offspring moves with a given mutation rate.

    Parameters:
    - bitboards: The current state of the board.
    - offspring: The list of offspring moves.
    - mutation_rate: The probability of mutating each move.
    - player: The current player ('r' for red, 'b' for blue).

    Returns:
    - A list of mutated offspring moves.
    """
    all_possible_moves = calculate_all_possible_moves(bitboards, player)
    move_list = [(start_pos, end_pos) for start_pos in all_possible_moves for end_pos in all_possible_moves[start_pos]]
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            offspring[i] = random.choice(move_list)
    return offspring


def evolutionary_algorithm_with_alpha_beta(bitboards, player, pop_size, num_generations, mutation_rate, move_time):
    """
    Executes an evolutionary algorithm to find the best move, using Alpha-Beta pruning for fitness evaluation.

    Parameters:
    - bitboards: The current state of the board.
    - player: The current player ('r' for red, 'b' for blue).
    - pop_size: The size of the population.
    - num_generations: The number of generations to simulate.
    - mutation_rate: The mutation rate for generating offspring.
    - move_time: The time allocated for evaluating each move.

    Returns:
    - A tuple of (best_move, best_fitness) where:
        - best_move: The best move found.
        - best_fitness: The fitness score of the best move.
    """
    population = generate_initial_population(bitboards, player, pop_size)

    for generation in range(num_generations):
        fitnesses = [fitness_function_alpha_beta(bitboards, player, move, move_time) for move in population]
        parents = select_parents(population, fitnesses, pop_size // 2)
        offspring = crossover(parents, pop_size - len(parents))
        population = parents + mutate(bitboards, offspring, mutation_rate, player)

        best_fitness = max(fitnesses) if player == 'r' else min(fitnesses)
        best_move = population[fitnesses.index(best_fitness)]

    best_fitness = max(fitnesses) if player == 'r' else min(fitnesses)
    best_move = population[fitnesses.index(best_fitness)]
    return best_move, best_fitness


def simulate_game_with_evolution_and_alpha_beta(fen, total_time=120, pop_size=5, num_generations=3, mutation_rate=0.1):
    """
    Simulates a game using a combination of evolutionary algorithms and Alpha-Beta pruning.

    Parameters:
    - fen: The FEN string representing the board state.
    - total_time: The total time allocated for the game simulation.
    - pop_size: The size of the population for the evolutionary algorithm.
    - num_generations: The number of generations to simulate.
    - mutation_rate: The mutation rate for generating offspring.

    Returns:
    - None
    """
    bitboards = parse_fen(reformulate(fen))
    player = 'b'
    start_time = time.time()

    while time.time() - start_time < total_time:
        if check_game_end(bitboards):
            break

        move_time = min(total_time - (time.time() - start_time), 1.0)
        best_move, _ = evolutionary_algorithm_with_alpha_beta(bitboards, player, pop_size, num_generations,
                                                              mutation_rate, move_time)

        if not best_move:
            print("No valid move found", player, "Lost")
            break

        start_pos, end_pos = best_move

        bitboards = do_move(Pos(*start_pos), Pos(*end_pos), player, bitboards)
        print(
            f"{player.upper()} moved from {Pos(*start_pos).to_chess_notation()} to {Pos(*end_pos).to_chess_notation()}")
        bitboards.print_combined_board()

        player = 'b' if player == 'r' else 'r'

    print("Game Over")
    game_end_status = check_game_end(bitboards)
    if game_end_status:
        print(game_end_status)
    else:
        print("Time limit reached")


# Example of using the simulation function
fen = "b0b0b0b0b0b0/1b0b0b0b0b0b01/8/8/8/8/1r0r0r0r0r0r01/r0r0r0r0r0r0"
simulate_game_with_evolution_and_alpha_beta(fen)
