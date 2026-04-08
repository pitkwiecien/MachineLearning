from typing import Dict, Iterable, List, Optional, Tuple

from two_player_games.game import Game
from two_player_games.move import Move
from two_player_games.player import Player
from two_player_games.state import State


class IsolationMove(Move):
    """Class that represents a move in the Isolation game.

    It has 1 numerical field:
     - field - represents the board index to which the player moves.
    """

    def __init__(self, field: int) -> None:
        self.field = field
        super().__init__()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, IsolationMove):
            return False
        return self.field == o.field


class IsolationState(State):
    """Represents a state in the Isolation game."""

    def __init__(
            self, size: int, current_player: Player, other_player: Player,
            grid: Optional[List[Optional[Player]]] = None,
            current_positions: Optional[Dict[Player, int]] = None) -> None:

        self.size = size

        if grid is None or current_positions is None:
            # Initialize a new board
            grid = [None for _ in range(size * size)]

            # Start at top-left and bottom-right corners
            start_pos_current = 0
            start_pos_other = (size * size) - 1

            grid[start_pos_current] = current_player
            grid[start_pos_other] = other_player

            current_positions = {
                current_player: start_pos_current,
                other_player: start_pos_other
            }

        self.grid = grid
        self.current_positions = current_positions

        super().__init__(current_player, other_player)

        self.finished, self.winner = self.check_finished()

    def check_finished(self) -> Tuple[bool, Optional[Player]]:
        """
        Checks if the current player has any valid adjacent moves.
        If they are trapped, they lose and the other player wins.
        """
        pos = self.current_positions[self._current_player]
        row, col = pos // self.size, pos % self.size

        has_moves = False

        # Check all 8 neighboring directions
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                r, c = row + dr, col + dc

                # Check bounds
                if 0 <= r < self.size and 0 <= c < self.size:
                    idx = r * self.size + c
                    # Check if space is unoccupied
                    if self.grid[idx] is None:
                        has_moves = True
                        break
            if has_moves:
                break

        if not has_moves:
            return True, self._other_player

        return False, None

    def get_moves(self) -> Iterable[IsolationMove]:
        moves: list[IsolationMove] = []
        if self.finished:
            return moves

        pos = self.current_positions[self._current_player]
        row, col = pos // self.size, pos % self.size

        # Look for valid moves in 8 directions
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                r, c = row + dr, col + dc

                if 0 <= r < self.size and 0 <= c < self.size:
                    idx = r * self.size + c
                    if self.grid[idx] is None:
                        moves.append(IsolationMove(idx))

        return moves

    def make_move(self, move: IsolationMove) -> 'IsolationState':
        if self.finished:
            raise ValueError("Cannot make move on a finished game.")

        if self.grid[move.field] is not None:
            raise ValueError("Cannot move to an occupied field.")

        # Ensure the move is to an adjacent square
        pos = self.current_positions[self._current_player]
        row, col = pos // self.size, pos % self.size
        m_row, m_col = move.field // self.size, move.field % self.size

        if abs(row - m_row) > 1 or abs(col - m_col) > 1:
            raise ValueError("Move must be to a neighboring field.")

        new_grid = list(self.grid)
        new_grid[move.field] = self._current_player

        new_positions = dict(self.current_positions)
        new_positions[self._current_player] = move.field

        return IsolationState(
            self.size, self._other_player, self._current_player,
            new_grid, new_positions
        )

    def is_finished(self) -> bool:
        return self.finished

    def get_winner(self) -> Optional[Player]:
        return self.winner

    def __str__(self) -> str:
        lines = []
        for r in range(self.size):
            row_str = []
            for c in range(self.size):
                idx = r * self.size + c
                field = self.grid[idx]

                # Highlight the current heads of the trails with brackets
                if self.current_positions[self._current_player] == idx:
                    row_str.append(f"[{field.char}]")
                elif self.current_positions[self._other_player] == idx:
                    row_str.append(f"[{field.char}]")
                else:
                    char = field.char if field is not None else ' '
                    row_str.append(f" {char} ")
            lines.append("|".join(row_str))

        divider = "\n" + ("-" * (self.size * 4 - 1)) + "\n"
        board_str = divider.join(lines)

        if self.is_finished():
            if self.get_winner() is None:
                status = "\nDraw!"
            else:
                status = f"\nWinner: Player {self.get_winner().char}"
        else:
            status = f"\nCurrent player: {self._current_player.char}"

        return "\n" + board_str + "\n" + status


class Isolation(Game):
    """Class that represents the game of Isolation."""
    FIRST_PLAYER_DEFAULT_CHAR = 'O'
    SECOND_PLAYER_DEFAULT_CHAR = 'X'

    def __init__(self, size: int = 7, first_player: Player = None, second_player: Player = None) -> None:
        """
        Initializes an Isolation game.

        Args:
            size (int): Dimensions of the NxN grid. Defaults to 7.
            first_player (Player): Player 1 object.
            second_player (Player): Player 2 object.
        """
        if first_player is None:
            first_player = Player(self.FIRST_PLAYER_DEFAULT_CHAR)
        if second_player is None:
            second_player = Player(self.SECOND_PLAYER_DEFAULT_CHAR)

        state = IsolationState(size, first_player, second_player)

        super().__init__(state)
