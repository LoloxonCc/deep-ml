import numpy as np

def det_2x2(matrix: list[list[int|float]]) -> float:
	return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def delete(matrix: list[list[int|float]], i: int) -> list[list[int|float]]:
	return [
	[val for idx_col, val in enumerate(row) if idx_col != 0]
	for idx_row, row in enumerate(matrix) if idx_row != i
	]

def det(matrix: list[list[int|float]]) ->float:
	if len(matrix) == 1:
		return matrix[0][0]
	
	if len(matrix) == 2:
		return det_2x2(matrix)
	
	return sum(((-1) ** i) * matrix[i][0] * det(delete(matrix, i)) for i in range(len(matrix)))

def determinant_4x4(matrix: list[list[int|float]]) -> float:
	return det(matrix)