#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

int GCD(int p_A, int p_B) {
	int c = p_A % p_B;

	while (c > 0)
	{
		p_A = p_B;
		p_B = c;
		c = p_A % p_B;
	}

	return p_B;
}

void ReduceFraction(int p_AUp, int p_ADown, int& p_CUp, int& p_CDown) {
	int gcd = 1;

	if (p_ADown != 0)
		gcd = GCD(abs(p_AUp), abs(p_ADown));

	p_CUp = p_AUp / gcd;
	p_CDown = p_ADown / gcd;

	if (p_CUp < 0 && p_ADown < 0 || p_CUp >= 0 && p_ADown < 0) {
		p_CUp *= -1;
		p_CDown *= -1;
	}
}

void AddFraction(int p_AUp, int p_ADown, int p_BUp, int p_BDown, int& p_CUp, int& p_CDown) {
	int outUp, outDown;

	if (p_ADown == p_BDown) {
		outDown = p_ADown;
		outUp = p_AUp + p_BUp;
	}
	else {
		outDown = p_ADown * p_BDown;
		outUp = p_AUp * p_BDown + p_BUp * p_ADown;
	}

	ReduceFraction(outUp, outDown, p_CUp, p_CDown);
}

void SubtractFraction(int p_AUp, int p_ADown, int p_BUp, int p_BDown, int& p_CUp, int& p_CDown) {
	int outUp, outDown;

	if (p_ADown == p_BDown) {
		outDown = p_ADown;
		outUp = p_AUp - p_BUp;
	}
	else {
		outDown = p_ADown * p_BDown;
		outUp = p_AUp * p_BDown - p_BUp * p_ADown;
	}

	ReduceFraction(outUp, outDown, p_CUp, p_CDown);
}

void MultiplyFraction(int p_AUp, int p_ADown, int p_BUp, int p_BDown, int& p_CUp, int& p_CDown) {
	int outUp, outDown, gcd;

	outDown = p_ADown * p_BDown;
	outUp = p_AUp * p_BUp;

	ReduceFraction(outUp, outDown, p_CUp, p_CDown);
}

void DivideFraction(int p_AUp, int p_ADown, int p_BUp, int p_BDown, int& p_CUp, int& p_CDown) {
	int outUp, outDown, gcd;

	MultiplyFraction(p_AUp, p_ADown, p_BDown, p_BUp, outUp, outDown);

	ReduceFraction(outUp, outDown, p_CUp, p_CDown);
}

void PrintFraction(int p_AUp, int p_ADown) {
	if (p_ADown == 1)
		cout << p_AUp;
	else
		cout << p_AUp << '/' << p_ADown;
}

void PrintArray(int p_RowSize, int p_ColSize, int** p_Array, int* p_VarsRow = nullptr, int* p_VarsCol = nullptr) {
	cout << endl;
	if (p_VarsRow != nullptr) {
		cout << "  ";
		for (int col = 0; col < p_ColSize - 1; col++) {
			if (p_VarsRow[col] < p_ColSize)
				cout << "\tx" << p_VarsRow[col] << " ";
			else
				cout << "\ty" << p_VarsRow[col] - p_ColSize + 1 << " ";
		}
		cout << endl;
	}

	for (int row = 0; row < p_RowSize; row++) {
		if (row < p_RowSize - 1 && p_VarsRow != nullptr) {
			if (p_VarsCol[row] >= p_ColSize)
				cout << "y" << p_VarsCol[row] - p_ColSize + 1 << " ";
			else
				cout << "x" << p_VarsCol[row] << " ";
		}
		else
			cout << "  ";

		cout.precision(2);
		for (int col = 0; col < p_ColSize * 2; col += 2) {
			cout << '\t';
			PrintFraction(p_Array[row][col], p_Array[row][col + 1]);
		}
		cout.precision(5);
		cout << endl;
	}
}

void TransformArray(int p_MinValueUp, int p_MinValueDown, int* p_RowSize, int* p_ColSize, int**& p_Array) {
	cout << "offseting by ";
	PrintFraction(-p_MinValueUp, p_MinValueDown);
	cout << endl;

	*p_RowSize = *p_RowSize + 1;
	*p_ColSize = *p_ColSize + 1;
	int** solution = new int* [*p_RowSize];
	for (int i = 0; i < *p_RowSize; ++i)
		solution[i] = new int[*p_ColSize * 2];

	for (int row = 0; row < *p_RowSize - 1; row++) {
		for (int col = 0; col < (*p_ColSize - 1) * 2; col += 2) {
			SubtractFraction(p_Array[row][col], p_Array[row][col + 1], p_MinValueUp, p_MinValueDown, solution[row][col], solution[row][col + 1]);
		}
		solution[row][*p_ColSize * 2 - 1] = 1;
		solution[row][*p_ColSize * 2 - 2] = 1;
	}

	for (int col = 0; col < (*p_ColSize - 1) * 2; col += 2) {
		solution[*p_RowSize - 1][col] = -1;
		solution[*p_RowSize - 1][col + 1] = 1;
	}
	solution[*p_RowSize - 1][*p_ColSize * 2 - 1] = 1;
	solution[*p_RowSize - 1][*p_ColSize * 2 - 2] = 0;

	p_Array = solution;
}

bool BlandsRule(int* p_Row, int* p_Col, int p_RowSize, int p_ColSize, int** p_Array, int* p_VarsRow, int* p_VarsCol) {
	*p_Col = -1;
	int smallestColVariable = -1;
	for (int col = 0; col < (p_ColSize - 1) * 2; col += 2) {
		if (p_Array[p_RowSize - 1][col] < 0 && (p_VarsRow[col / 2] < smallestColVariable || smallestColVariable < 0)) {
			*p_Col = col / 2;
			smallestColVariable = p_VarsRow[col / 2];
		}
	}

	if (*p_Col < 0)
		return false;

	int smallestRowVariable = -1, smallestRatioUp, smallestRatioDown;
	*p_Row = 0;
	bool foundZeroRatio = false;
	for (int row = 0; row < p_RowSize - 1; row++) {
		if (p_Array[row][*p_Col * 2] <= 0)
			continue;

		if (p_Array[row][p_ColSize * 2 - 2] == 0 && (p_VarsCol[row / 2] < smallestRowVariable || smallestRowVariable < 0 || !foundZeroRatio)) {
			*p_Row = row;
			smallestRowVariable = p_VarsCol[row / 2];
			foundZeroRatio = true;
		}
		else if (!foundZeroRatio) {
			if (smallestRowVariable < 0) {
				DivideFraction(p_Array[row][p_ColSize * 2 - 2], p_Array[row][p_ColSize * 2 - 1], p_Array[row][*p_Col * 2], p_Array[row][*p_Col * 2 + 1], smallestRatioUp, smallestRatioDown);
				*p_Row = row;
				smallestRowVariable = p_VarsCol[row / 2];
			}
			else {
				int currentRatioUp, currentRatioDown;
				DivideFraction(p_Array[row][p_ColSize * 2 - 2], p_Array[row][p_ColSize * 2 - 1], p_Array[row][*p_Col * 2], p_Array[row][*p_Col * 2 + 1], currentRatioUp, currentRatioDown);
				SubtractFraction(currentRatioUp, currentRatioDown, smallestRatioUp, smallestRatioDown, currentRatioUp, currentRatioDown);

				if (currentRatioUp < 0) {
					*p_Row = row;
					smallestRowVariable = p_VarsCol[row / 2];
					smallestRatioUp = currentRatioUp;
					smallestRatioDown = currentRatioDown;
				}
				else if (currentRatioUp == 0 && p_VarsCol[row / 2] < smallestRowVariable) {
					*p_Row = row;
					smallestRowVariable = p_VarsCol[row / 2];
					smallestRatioUp = currentRatioUp;
					smallestRatioDown = currentRatioDown;
				}
			}
		}
	}

	cout << "row: " << *p_Row << ", col: " << *p_Col << endl;
	return true;
}

void EnterVariable(int p_Row, int p_Col, int*& p_VarsRow, int*& p_VarsCol) {
	int temp = p_VarsRow[p_Col];
	p_VarsRow[p_Col] = p_VarsCol[p_Row];
	p_VarsCol[p_Row] = temp;
}

void  PivotArray(int p_Row, int p_Col, int p_RowSize, int p_ColSize, int**& p_Array) {
	int** solution = new int* [p_RowSize];

	for (int row = 0; row < p_RowSize; row++) {
		solution[row] = new int[p_ColSize * 2];
		for (int col = 0; col < p_ColSize * 2; col += 2) {
			if (p_Row == row && p_Col == col / 2)
				DivideFraction(1, 1, p_Array[p_Row][p_Col * 2], p_Array[p_Row][p_Col * 2 + 1], solution[row][col], solution[row][col + 1]);
			else if (p_Row == row)
				DivideFraction(p_Array[row][col], p_Array[row][col + 1], p_Array[p_Row][p_Col * 2], p_Array[p_Row][p_Col * 2 + 1], solution[row][col], solution[row][col + 1]);
			else if (p_Col == col / 2) {
				DivideFraction(p_Array[row][col], p_Array[row][col + 1], p_Array[p_Row][p_Col * 2], p_Array[p_Row][p_Col * 2 + 1], solution[row][col], solution[row][col + 1]);
				solution[row][col] *= -1;
			}
			else {
				MultiplyFraction(p_Array[p_Row][col], p_Array[p_Row][col + 1], p_Array[row][p_Col * 2], p_Array[row][p_Col * 2 + 1], solution[row][col], solution[row][col + 1]);
				DivideFraction(solution[row][col], solution[row][col + 1], p_Array[p_Row][p_Col * 2], p_Array[p_Row][p_Col * 2 + 1], solution[row][col], solution[row][col + 1]);
				SubtractFraction(p_Array[row][col], p_Array[row][col + 1], solution[row][col], solution[row][col + 1], solution[row][col], solution[row][col + 1]);
			}
		}
	}

	p_Array = solution;
}

void GetResults(int minValueUp, int minValueDown, int p_RowSize, int p_ColSize, int** p_Array, int* p_VarsRow, int* p_VarsCol) {
	int vUp = p_Array[p_RowSize - 1][p_ColSize * 2 - 2];
	int vDown = p_Array[p_RowSize - 1][p_ColSize * 2 - 1];
	int* player1 = new int[(p_RowSize - 1) * 2];
	for (int i = 0; i < (p_RowSize - 1) * 2; i += 2) {
		player1[i] = 0;
		player1[i + 1] = 1;
	}

	for (int i = 0; i < (p_ColSize - 1) * 2; i += 2) {
		if (p_VarsRow[i / 2] >= p_ColSize) {
			int pIndex = p_VarsRow[i / 2] - p_ColSize;
			DivideFraction(p_Array[p_RowSize - 1][i], p_Array[p_RowSize - 1][i + 1], vUp, vDown, player1[pIndex * 2], player1[pIndex * 2 + 1]);
		}
	}

	cout << "p = ";
	if (p_RowSize > 1) {
		PrintFraction(player1[0], player1[1]);
		cout << endl;
		for (int i = 2; i < (p_RowSize - 1) * 2; i += 2) {
			cout << "    ";
			PrintFraction(player1[i], player1[i + 1]);
			cout << endl;
		}
	}

	int* player2 = new int[(p_ColSize - 1) * 2];
	for (int i = 0; i < (p_ColSize - 1) * 2; i += 2) {
		player2[i] = 0;
		player2[i + 1] = 1;
	}

	for (int i = 0; i < (p_RowSize - 1) * 2; i += 2) {
		if (p_VarsCol[i / 2] < p_ColSize) {
			int qIndex = p_VarsCol[i / 2] - 1;
			DivideFraction(p_Array[i / 2][p_ColSize * 2 - 2], p_Array[i / 2][p_ColSize * 2 - 1], vUp, vDown, player2[qIndex * 2], player2[qIndex * 2 + 1]);
		}
	}

	cout << "q = ";
	if (p_ColSize > 1) {
		PrintFraction(player2[0], player2[1]);
		cout << endl;
		for (int i = 2; i < (p_ColSize - 1) * 2; i += 2) {
			cout << "    ";
			PrintFraction(player2[i], player2[i + 1]);
			cout << endl;
		}
	}

	DivideFraction(1, 1, vUp, vDown, vUp, vDown);
	AddFraction(vUp, vDown, minValueUp, minValueDown, vUp, vDown);

	cout << "v = ";
	PrintFraction(vUp, vDown);
	cout << endl;
}

int FillSolution(ifstream& p_FileStream, int p_RowSize, int p_ColSize, int**& p_Array, int* p_MinUp, int* p_MinDown) {
	for (int row = 0; row < p_RowSize; row++) {
		for (int column = 0; column < p_ColSize * 2; column += 2) {
			p_FileStream >> p_Array[row][column];

			if (!p_FileStream) {
				cout << "Error reading file for element " << row << "," << column / 2 << endl;
				return 1;
			}

			if (p_FileStream.peek() == '/') {
				p_FileStream.get();
				p_FileStream >> p_Array[row][column + 1];
			}
			else
				p_Array[row][column + 1] = 1.0;

			if (!p_FileStream) {
				cout << "Error reading file for element " << row << "," << column / 2 << endl;
				return 1;
			}

			if (p_Array[row][column] / p_Array[row][column + 1] < *p_MinUp / *p_MinDown) {
				*p_MinUp = p_Array[row][column];
				*p_MinDown = p_Array[row][column + 1];
			}
		}
	}
	return 0;
}

int run_simplex(int argc, char** argv)
{
	ifstream fp("../Simplex.txt");
	if (!fp) {
		cout << "Error, file couldn't be opened" << endl;
		return 1;
	}

	int solutionsCount, rowCount, colCount;
	fp >> solutionsCount;
	for (int solutionsIndex = 0; solutionsIndex < solutionsCount; solutionsIndex++) {
		int minValueUp = 0, minValueDown = 1;
		fp >> rowCount;
		fp >> colCount;
		int** solution = new int* [rowCount];
		for (int i = 0; i < rowCount; i++)
			solution[i] = new int[colCount * 2];

		int* varsRow = new int[colCount];
		for (int i = 0; i < colCount; i++)
			varsRow[i] = i + 1;
		int* varsCol = new int[rowCount];
		for (int i = colCount; i < colCount + rowCount; i++)
			varsCol[i - colCount] = i + 1;

		if (FillSolution(fp, rowCount, colCount, solution, &minValueUp, &minValueDown))
			return 1;
		PrintArray(rowCount, colCount, solution);
		TransformArray(minValueUp, minValueDown, &rowCount, &colCount, solution);
		PrintArray(rowCount, colCount, solution, varsRow, varsCol);

		int col = -1, row = -1;
		while (BlandsRule(&row, &col, rowCount, colCount, solution, varsRow, varsCol)) {
			PivotArray(row, col, rowCount, colCount, solution);
			EnterVariable(row, col, varsRow, varsCol);
			PrintArray(rowCount, colCount, solution, varsRow, varsCol);
		}

		GetResults(minValueUp, minValueDown, rowCount, colCount, solution, varsRow, varsCol);
		cout << "-----" << endl;
	}

	return 0;
}