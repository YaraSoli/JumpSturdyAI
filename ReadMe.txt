JumpSturdy Game Simulator
This project simulates a custom board game using evolutionary algorithms and Alpha-Beta pruning to optimize move decisions for two players ('r' and 'b'). The game operates on a bitboard-based system and can simulate entire games within a predefined time limit.

Features:
Bitboard Representation: Efficient storage and manipulation of game states using bitboards.
Evolutionary Algorithm: Simulates multiple generations of moves to optimize the decision-making process for each player.
Alpha-Beta Pruning: A minimax search algorithm to reduce the number of nodes evaluated in the game tree.
Game Simulation: Full game simulation with time limits, and move evaluation.
Customizable Settings: Population size, mutation rate, and number of generations can be configured for the evolutionary algorithm.

AI Technologies Used:
Transposition Tables: Used to store previously evaluated board positions during the Alpha-Beta pruning process, reducing the need to re-evaluate positions that have already been computed. This improves the efficiency of the search algorithm by avoiding redundant calculations.

Alpha-Beta Pruning: A search algorithm that minimizes the number of nodes evaluated in a decision tree by pruning branches that cannot influence the final decision. This is combined with the evolutionary algorithm to optimize move selection for both players.

Evolutionary Algorithm: A genetic algorithm-based approach that generates a population of potential moves, evaluates their fitness, and evolves them over several generations to find the optimal strategy. Moves are mutated and crossed over to create new generations, improving the decision-making process over time.

Dynamic Time Management: The simulation allocates a fixed total time for the game, with each move having a dynamically adjusted time limit. This ensures that both players make decisions within the available game time, optimizing their use of computational resources.

Piece Value Heuristics: The score for each move is evaluated based on predefined values assigned to pieces, such as normal pieces and promoted pieces (e.g., r, b, rr, bb). This evaluation also includes a bonus for capturing opponent pieces, helping guide the evolutionary algorithm towards more favorable moves.
