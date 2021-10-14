// cv01_threads.cpp: Definuje vstupn� bod pro konzolovou aplikaci.
//

#include "stdafx.h"
#include <iostream>

#include <thread>
#include <future>
#include <chrono>
#include <random>

#include <vector>
#include <algorithm>
#include <utility>

#include <sstream>
#include <string>

#include "cClientSocket.h"

using namespace std;

typedef vector<double> t_point;

/**
* Create a PRNG for every thread.
* TODO: make a safe initialization, use this_thread::get_id()
*/
double drand() {
	static thread_local mt19937 generator;
	uniform_int_distribution<unsigned int> distribution(0, 10);
	return distribution(generator);
}


string download(int index)
{
#if defined WIN32
	WSADATA wsa_data;
	WSAStartup(MAKEWORD(1, 1), &wsa_data);
#endif

	//URL: http://name-service.appspot.com/api/v1/names/0.xml

	string host = "name-service.appspot.com";
	stringstream ss;
	ss << "/api/v1/names/" << index << ".xml";
	string addr = ss.str();

	string get_http = "GET " + addr + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";

	arg::cClientSocket client;
	if (!client.OpenClientSocket(host, 80)) {
		cout << "Not connected!" << endl;
		return 0;
	}

	client.Send(get_http);
	string response = client.Receive();
	//cout << "Response: " << endl << endl << response << endl;
	client.Close();

#if defined WIN32
	WSACleanup();
#endif

	return response;
}


/**
* Generate CNT random points of DIM dimensions. Single thread.
*/
vector<t_point> my_gen(unsigned int dim, unsigned int cnt)
{
	vector<t_point> results;

	t_point p;
	p.resize(dim);

	for (unsigned int i = 0; i < cnt; i++)
	{
		for (unsigned int j = 0; j < dim; j++)
			p[j] = drand();
		results.push_back(p);
	}

	return results;
}

vector<string> my_gen2(unsigned int start, unsigned int length)
{
	vector<string> results;

	for (unsigned int i = start; i < start+length; i++)
	{
		results.push_back(download(i));
	}

	return results;
}

/**
* Helper function that just prints stuff.
*/
void print(vector<t_point> & data)
{
	for (t_point item : data)
	{
		for (double d : item)
			cout << d << "\t";
		cout << endl;
	}
}

void print(vector<string> & data)
{
	for (string item : data)
	{
		for (char d : item)
			cout << d;
		cout << endl;
	}
}

/**
* Experimental function that:
*  Generates CNT random DIM-dimensional points
*
* This all will be done in THREADS threads.
* Note 1: launch::async is not required for MSVC, but must be used for gcc, otherwise it will allways run in single thread.
*
*/
void experiment(const unsigned int DIM, const unsigned int CNT, const unsigned int THREADS)
{
	chrono::time_point<chrono::system_clock> start, end;
	start = chrono::system_clock::now();

	vector<future<vector<t_point>>> my_futures;

	for (unsigned int i = 0; i < THREADS; i++)
	{
		my_futures.push_back(async(launch::async, my_gen, DIM, CNT / THREADS));
	}

	vector<t_point> my_points;
	vector<t_point> current;

	for (unsigned int i = 0; i < my_futures.size(); i++)
	{
		current = my_futures[i].get();
		my_points.insert(my_points.end(), current.begin(), current.end());
	}

	end = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end - start;

	if (CNT < 40)
		print(my_points);

	cout << "Data " << CNT << ", Threads " << THREADS << ", Elapsed time " << elapsed_seconds.count() << "s\n";
}

void experiment2(const unsigned int CNT, const unsigned int THREADS)
{
	chrono::time_point<chrono::system_clock> start, end;
	start = chrono::system_clock::now();

	vector<future<vector<string>>> my_futures;

	for (unsigned int i = 0; i < THREADS; i++)
	{
		int length = CNT / THREADS;
		my_futures.push_back(async(launch::async, my_gen2, i*length, length));
	}

	vector<string> my_points;
	vector<string> current;

	for (unsigned int i = 0; i < my_futures.size(); i++)
	{
		current = my_futures[i].get();
		my_points.insert(my_points.end(), current.begin(), current.end());
	}

	end = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end - start;

	if (CNT < 40)
		print(my_points);

	cout << "Data " << my_points.size() << ", Threads " << THREADS << ", Elapsed time \t" << elapsed_seconds.count() << "s, each \t" << elapsed_seconds.count()/my_points.size()*1000 << "ms\n";
}

int main()
{	
	/*experiment(100, 1000000, 1);
	cout << endl;
	cout << endl;
	experiment(100, 1000000, 4);*/

	/*experiment2(200, 1);
	cout << endl;
	cout << endl;
	experiment2(200, 5);*/

	for (unsigned int i = 20; i <= 150; i++)
		experiment2(500, i);


	return 0;
}