import random
import numpy as np

from puzzle_generator import PuzzleGenerator

class AdvancedSudokuGenerator(PuzzleGenerator):
    
    def generate_professional_sudoku(self, min_clues=30, symmetry=False, required_difficulty="medium"):
        grid = np.zeros((9, 9), dtype=int)
        self.fill_grid(grid)  # Create a fully solved grid

        if symmetry:
            puzzle = self.remove_numbers_with_symmetry(grid.copy(), num_clues=min_clues)
        else:
            puzzle = self.remove_numbers_exact_clues(grid.copy(), num_clues=min_clues)

        # Ensure the puzzle has exactly the desired number of clues
        puzzle = self.enforce_exact_clue_count(puzzle, min_clues)

        return puzzle, grid

    # Remove numbers symmetrically from the grid
    def remove_numbers_with_symmetry(self, grid, num_clues):
        total_cells = 81
        cells_to_remove = total_cells - num_clues
        removed = 0
        
        # Get all symmetric pairs
        symmetric_pairs = [(r, c, 8 - r, 8 - c) for r in range(9) for c in range(9) if r <= 8 - r and c <= 8 - c]
        random.shuffle(symmetric_pairs)

        for r1, c1, r2, c2 in symmetric_pairs:
            if removed >= cells_to_remove // 2:
                break

            if grid[r1][c1] == 0 or grid[r2][c2] == 0:
                continue

            backup1, backup2 = grid[r1][c1], grid[r2][c2]
            grid[r1][c1], grid[r2][c2] = 0, 0

            # Ensure unique solution
            if self.has_unique_solution(grid):
                removed += 2  # Removing two cells symmetrically
            else:
                grid[r1][c1], grid[r2][c2] = backup1, backup2  # Restore if removing breaks uniqueness

        return grid

    # Remove numbers to leave exactly num_clues in the grid
    def remove_numbers_exact_clues(self, grid, num_clues):
        total_cells = 81
        cells_to_remove = total_cells - num_clues
        removed = 0

        all_cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(all_cells)

        for row, col in all_cells:
            if removed >= cells_to_remove:
                break

            if grid[row][col] == 0:
                continue

            backup = grid[row][col]
            grid[row][col] = 0

            # Check if the puzzle still has a unique solution
            if self.has_unique_solution(grid):
                removed += 1  # Successful removal
            else:
                grid[row][col] = backup  # Restore if removing breaks uniqueness

        return grid

    # Enforce the exact clue count
    def enforce_exact_clue_count(self, grid, min_clues):
        """
        Ensure the puzzle has exactly `min_clues` by forcefully removing or restoring cells.
        """
        current_clues = np.count_nonzero(grid)
        total_cells = 81

        # If too many cells removed, restore some cells
        if current_clues < min_clues:
            # Get all removed cells and restore randomly until exactly `min_clues`
            all_cells = [(r, c) for r in range(9) for c in range(9) if grid[r][c] == 0]
            random.shuffle(all_cells)
            for row, col in all_cells:
                if current_clues >= min_clues:
                    break
                grid[row][col] = self.solution[row][col]  # Restore from the solution
                current_clues += 1

        # If too few cells removed, remove more until exactly `min_clues`
        if current_clues > min_clues:
            all_filled_cells = [(r, c) for r in range(9) for c in range(9) if grid[r][c] != 0]
            random.shuffle(all_filled_cells)
            for row, col in all_filled_cells:
                if current_clues <= min_clues:
                    break
                grid[row][col] = 0  # Remove
                current_clues -= 1

        return grid
