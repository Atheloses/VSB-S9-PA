// CV02.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;

float distance(int x1, int y1, int x2, int y2)
{
	// Calculating distance
	return sqrt(pow(x2 - x1, 2) +
		pow(y2 - y1, 2) * 1.0);
}

char to_letter(int n)
{
	return static_cast<char>('A' - 1 + n);
}

int to_number(char n)
{
	return n - 'A';
}

float permutace(vector<vector<float>> matice, string perm) {
	float sum = 0;
	int minuleMesto = to_number(perm[0]);
	for (int i = 1; i < perm.size(); i++)
	{
		int noveMesto = to_number(perm[i]);
		sum += matice[minuleMesto][noveMesto];
		minuleMesto = noveMesto;
	}
	sum += matice[minuleMesto][to_number(perm[0])];
	return sum;
}

inline bool file_exists(const std::string& name) {
	ifstream f(name.c_str());
	return f.good();
}

int fact(int n) {
	if (n == 0 || n == 1)
		return 1;
	else
		return n * fact(n - 1);
}

//void tss_threads(const unsigned int CNT, const unsigned int THREADS)
//{
//	chrono::time_point<chrono::system_clock> start, end;
//	start = chrono::system_clock::now();
//
//	vector<future<vector<string>>> my_futures;
//
//	for (unsigned int i = 0; i < THREADS; i++)
//	{
//		int length = CNT / THREADS;
//		my_futures.push_back(async(launch::async, my_gen2, i * length, length));
//	}
//
//	vector<string> my_points;
//	vector<string> current;
//
//	for (unsigned int i = 0; i < my_futures.size(); i++)
//	{
//		current = my_futures[i].get();
//		my_points.insert(my_points.end(), current.begin(), current.end());
//	}
//
//	end = chrono::system_clock::now();
//	chrono::duration<double> elapsed_seconds = end - start;
//
//	if (CNT < 40)
//		print(my_points);
//
//	cout << "Data " << my_points.size() << ", Threads " << THREADS << ", Elapsed time \t" << elapsed_seconds.count() << "s, each \t" << elapsed_seconds.count() / my_points.size() * 1000 << "ms\n";
//}

int main(int argc, char** argv)
{
	string path = "C:\\Users\\PUS0065\\Downloads\\CV02\\CV02\\x64\\Release\\problem7.txt";

	if (argc > 1)
		path = argv[1];

	ifstream infile(path);

	if (!file_exists(path)) {
		cout << "file does not exists" << endl;
		return 0;
	}

	string line;
	vector<vector<float>> mesta;

	while (getline(infile, line))
	{
		istringstream iss(line);
		int a;
		float b, c;
		if ((iss >> a >> b >> c))
		{
			cout << a << ", " << b << ", " << c << endl;
			vector<float> noveMesto;
			noveMesto.push_back(a);
			noveMesto.push_back(b);
			noveMesto.push_back(c);
			mesta.push_back(noveMesto);
		}

		// process pair (a,b)
	}
	vector<vector<float>> matice;

	for (int i = 0; i < mesta.size(); i++)
	{
		vector<float> novyRadek;
		for (int j = 0; j < mesta.size(); j++)
		{
			novyRadek.push_back(distance(mesta[i][1], mesta[i][2], mesta[j][1], mesta[j][2]));
		}
		matice.push_back(novyRadek);
	}

	string s;
	for (int i = 0; i < matice.size(); i++) {
		for (int j = 0; j < matice.size(); j++) {
			printf("%6.3f, ", matice[i][j]);
		}
		cout << endl;
		s += to_letter(i + 1);
	}

	int len = s.size();
	float minPerm = permutace(matice, s);
	unsigned long long tries = 0;
	string best = s;
	//#pragma omp parallel for reduction(min:minPerm)
	do {
		//cout << s << endl;
		float newPerm = permutace(matice, s);
		tries++;
		if (newPerm < minPerm) {
			best = s;
			minPerm = newPerm;
		}
	} while (next_permutation(s.begin()+1, s.end()));

	cout << "Best: ";
	for (int i = 0; i < best.size(); i++)
		cout << to_number(best[i]) + 1;
	cout << ", with " << minPerm << ", out of " << tries << " tries" << endl;
	unsigned long long perms = fact(len) / fact(len - len);
	cout << "permutations: " << perms;

	/*string output;
	cin >> output;*/
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
