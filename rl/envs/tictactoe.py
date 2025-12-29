"""Tic-Tac-Toe Environment"""
import numpy as np
from rl.envs.base import BaseEnvironment

class TicTacToeEnv(BaseEnvironment):
  """Tic-Tac-Toe game"""

  def __init__(self):
    super().__init__()
    self.reset()

  def reset(self):
    self.board = np.zeros((3, 3), dtype=int)
    self.current_player = 1
    self.done = False
    self.winner = None
    return self._get_state()

  def _get_state(self):
    board_enc = np.where(self.board.flatten() == -1, 2, self.board.flatten())
    return sum(int(v) * (3 ** i) for i, v in enumerate(board_enc))

  def _check_winner(self):
    lines = list(self.board) + list(self.board.T) + \
            [self.board.diagonal(), np.fliplr(self.board).diagonal()]
    for line in lines:
      if abs(line.sum()) == 3:
        return line[0]
    return 0 if not (self.board == 0).any() else None

  def step(self, a):
    if self.done:
      return self._get_state(), 0, True, {}

    r, c = divmod(a, 3)
    if self.board[r, c] != 0:
      return self._get_state(), -1, False, {'invalid_move': True}

    self.board[r, c] = self.current_player
    result = self._check_winner()
    if result is not None:
      self.done, self.winner = True, result
      return self._get_state(), (1 if result == 1 else -1 if result == -1 else 0.5), True, {}

    self.current_player = -1
    valid = self.get_valid_actions()
    if valid:
      r, c = divmod(np.random.choice(valid), 3)
      self.board[r, c] = -1
      result = self._check_winner()
      if result is not None:
        self.done, self.winner = True, result
        return self._get_state(), (-1 if result == -1 else 0.5), True, {}

    self.current_player = 1
    return self._get_state(), 0, False, {}

  def render(self, mode='rgb_array'):
    cell, grid = 100, 300
    img = np.ones((grid, grid, 3), dtype=np.uint8) * 255

    for i in range(4):
      img[i*cell:i*cell+3, :] = 0
      img[:, i*cell:i*cell+3] = 0

    for i in range(3):
      for j in range(3):
        cy, cx = i*cell+50, j*cell+50
        if self.board[i, j] == 1:
          for k in range(-35, 36):
            img[cy+k, cx+k] = img[cy+k, cx-k] = [102, 126, 234]
        elif self.board[i, j] == -1:
          for dy in range(-35, 36):
            for dx in range(-35, 36):
              if 30**2 < dx*dx + dy*dy < 35**2:
                img[cy+dy, cx+dx] = [240, 147, 251]

    if self.done:
      banner = np.ones((40, grid, 3), dtype=np.uint8) * \
               ([102,126,234] if self.winner==1 else
                [240,147,251] if self.winner==-1 else [150]*3)
      img = np.vstack([img, banner])

    return img

  def get_state_space_size(self):
    return 3 ** 9

  def get_action_space_size(self):
    return 9

  def get_valid_actions(self):
    return [i for i in range(9) if self.board[i//3, i%3] == 0]
