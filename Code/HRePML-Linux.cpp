#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <float.h>
#include "time.h"
#include <boost/math/differentiation/finite_difference.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#include "lbfgs.h"

extern "C" {
	void dgetrf_(const int* M, const int* N, double* A, const int* LDA, 
              const int* IPIV, const int* INFO);

 	void dgetri_(const int* N, double* A, const int* LDA, const int* IPIV,
              double* WORK, const int* LWORK, const int* INFO);

	void dgeev_(char* JOBVL, char* JOBVR, const int* N, double* A, const int* LDA, double* WR, double* WI,
			  double* VL, const int* LDVL, double* VR, const int* LDVR, double* WORK, const int* LWORK, const int* INFO);
}

#define NLbf 100

typedef unsigned int   uint;
typedef unsigned char  uchar;
typedef unsigned short ushort;

using namespace std;

static double rlmmPvalue = 1e-2;
static uint s;
static vector<vector<double>> delta;
static vector<vector<double>> xu;
static vector<vector<double>> yu;

static double yy1, yx1, xx1, zx, zy, zz;
static vector<vector<double>> h1;

vector<vector<double>> ReadFile(string filename)
{
	uint totalnum = 0; // total num of elements in matrix
	uint rownum = 0;
	uint colnum = 0;
	//clock_t start = clock();
	ifstream fin(filename, std::ios::binary);
	if (fin.fail()) {
		cerr << "\nError:" << filename << " could not be opened.\n\n";
		exit(1);
	}
	vector<char> buf(static_cast<unsigned int>(fin.seekg(0, std::ios::end).tellg()));
	fin.seekg(0, std::ios::beg).read(&buf[0], static_cast<std::streamsize>(buf.size()));
	fin.close();
	//clock_t end = clock();
	//std::cout << "time : " << ((double)end - start) / CLOCKS_PER_SEC << "s\n";
	vector<char>::iterator it;
	vector<double> data;
	string next = ""; // generally read in chars
	for (it = buf.begin(); it != buf.end(); it++)
	{
		if (it[0] == ',' || it[0] == '\n') {    // end of value
			data.push_back(atof(next.c_str())); // add value
			next = "";                          // clear
			if (it[0] == '\n') {
				rownum = rownum + 1;
			}
			totalnum = totalnum + 1;
		}
		else {
			next += it[0];                      // add to read string
		}
	}
	colnum = totalnum / rownum;
	if (colnum != uint(colnum))
	{
		cerr << "\nError:" << filename << " maybe not a matrix format.\n\n";
		exit(1);
	}
	cout << rownum << " " << colnum << " " << totalnum << endl;
	cout << "\n\n";
	vector<vector<double>> MatrixData(rownum, vector<double>(colnum));
	for (uint i = 0; i < rownum; i++) {
		for (uint j = 0; j < colnum; j++) {
			MatrixData[i][j] = data[i * colnum + j];
		}
	}
	return MatrixData;
}

vector<vector<double>> MatrixMultiply(vector<vector<double>> arrA, vector<vector<double>> arrB)
{
	uint rowA = arrA.size();
	uint colA = arrA[0].size();
	uint rowB = arrB.size();
	uint colB = arrB[0].size();
	vector<vector<double>>  res;
	if (colA != rowB)
	{
		return res;
	}
	else
	{
		res.resize(rowA);
		for (uint i = 0; i < rowA; ++i)
		{
			res[i].resize(colB);
		}
		for (uint i = 0; i < rowA; ++i)
		{
			for (uint j = 0; j < colB; ++j)
			{
				for (uint k = 0; k < colA; ++k)
				{
					res[i][j] += arrA[i][k] * arrB[k][j];
				}
			}
		}
	}
	return res;
}

vector<vector<double>> Transpose(vector<vector<double>> matrix)
{
	vector<vector<double>>v(matrix[0].size(), vector<double>());
	for (uint i = 0; i < matrix.size(); i++)
	{
		for (uint j = 0; j < matrix[0].size(); j++)
		{
			v[j].push_back(matrix[i][j]);
		}
	}
	return v;
}

vector<vector<double>> MatrixInverse(vector<vector<double>> matrix)
{
	int N = matrix.size();
	int info = 0;
	int lworkspace = N;
	int *ipiv = new int[N];
	double* A = new double[N * N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i * N + j] = matrix[i][j];
		}
	}
	dgetrf_(&N, &N, A, &N, ipiv, &info);
	double* workspace = new double[lworkspace*sizeof(double)];
	dgetri_(&N, A, &N, ipiv, workspace, &lworkspace, &info);
  	delete [] ipiv;
  	delete [] workspace;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrix[i][j] = A[i * N + j];
		}
	}
	delete A;
	return matrix;
}

vector<vector<double>> Eigvecval(vector<vector<double>> matrix)
{
	int N = matrix.size();
	char jobvl = 'V';
	char jobvr = 'V';
	int lda = N;
	int ldvr = N;
	int ldvl = N;
	int lwork = N * 4;
	int info = 0;
	double* wr = new double[N];
	double* wi = new double[N];
	double* vr = new double[N * ldvr];
	double* vl = new double[N * ldvl];
	double* work = new double[N * lwork];
	double* A = new double[N * N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i * N + j] = matrix[i][j];
		}
	}
	dgeev_(&jobvl, &jobvr, &N, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);

	delete wr;
	delete wi;
	delete vl;
	delete work;
	vector<vector<double>> eigvecval((N + 1), vector<double>(N));
	for (int i = 0; i < N; i++) {        //N*N eigenvectors
		for (int j = 0; j < N; j++) {
			eigvecval[i][j] = vr[i * N + j];
		}
	}
	for (int i = 0; i < N; i++)    // last row of eigvecval is the eigenvalues
	{
		eigvecval[N][i] = A[i * N + i];
	}
	delete A;
	delete vr;
	return eigvecval;
}

double sumlog(vector<vector<double>> arrA, double lambda)
{
	uint rowA = arrA.size();
	double sumlogval = 0.0;
	for (uint i = 0; i < rowA; ++i)
	{
		sumlogval = sumlogval + log(arrA[i][0] * lambda + 1);
	}
	return sumlogval;
}

double loglikeInner(double theta)
{
	double lambda = exp(theta);
	double logdt = sumlog(delta, lambda);
	double yy = 0.0, yx = 0.0, xx = 0.0;
	double loglike;
	int N = delta.size();
	vector<double> h(N);

	for (int i = 0; i < N; i++) {
		h[i] = 1 / (lambda * delta[i][0] + 1);
		yy = yy + yu[i][0] * h[i] * yu[i][0];
		yx = yx + yu[i][0] * h[i] * xu[i][0];
		xx = xx + xu[i][0] * h[i] * xu[i][0];
	}
	loglike = -0.5 * logdt - 0.5 * (N - s) * log(yy - yx * (1 / xx) * yx) - 0.5 * log(xx);
	return -loglike;
}

vector<double> fixedInner(double lambda)
{
	int N = delta.size();
	double yy = 0.0, yx = 0.0, xx = 0.0;
	double sigma2, stderror, beta;
	vector<double> h(N);
	vector<double> fixres(3);

	for (int i = 0; i < N; i++) {
		h[i] = 1 / (lambda * delta[i][0] + 1);
		yy = yy + yu[i][0] * h[i] * yu[i][0];
		yx = yx + yu[i][0] * h[i] * xu[i][0];
		xx = xx + xu[i][0] * h[i] * xu[i][0];
	}

	beta = (1 / xx) * yx;
	sigma2 = (yy - yx * (1 / xx) * yx) / (N - s);
	stderror = sqrt((1 / xx) * sigma2);
	fixres[0] = beta;
	fixres[1] = stderror;
	fixres[2] = sigma2;

	return fixres;
}

static lbfgsfloatval_t evaluate1(
	void* instance,
	const lbfgsfloatval_t* thetaLbf,
	lbfgsfloatval_t* gLbf,
	const int n,
	const lbfgsfloatval_t step
)
{
	lbfgsfloatval_t loglikeInnerLbf = 0.0;
	for (int i = 0; i < n; i++) {
		if ((thetaLbf[i] > -50) && (thetaLbf[i] < 10)) {
			gLbf[i] = boost::math::differentiation::finite_difference_derivative(loglikeInner, thetaLbf[i]);
			loglikeInnerLbf = loglikeInner(thetaLbf[i]);
		}
	}
	return loglikeInnerLbf;
}

static int progress1(
	void* instance,
	const lbfgsfloatval_t* thetaLbf,
	const lbfgsfloatval_t* gLbf,
	const lbfgsfloatval_t loglikeInnerLbf,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step,
	int n,
	int k,
	int ls
)
{
	//cout << "Iteration:" << k << endl;
	//cout << "loglikeInnerLbf = " << loglikeInnerLbf << " " << "thetaLbf = " << thetaLbf[0] << endl;
	//cout << "xnorm = " << xnorm << " " << "gnorm = " << gnorm << " " << "step = " << step << endl;
	//cout << endl;
	return 0;
}

vector<double> mixed(vector<vector<double>> X, vector<vector<double>> Y, vector<vector<double>> KK)
{
	int ret;
	lbfgsfloatval_t loglikeInnerLbf;
	lbfgsfloatval_t* thetaLbf = lbfgs_malloc(NLbf);
	lbfgs_parameter_t param;
	double lambda, fn0, fn1, lrt, lrt0, lod, pvalue, sigma2g;
	vector<double> parmfix;
	vector<double> parameter(8);

	if (thetaLbf == NULL) {
		cerr << "ERROR: Failed to allocate a memory block for variables.\n" << endl;
	}
	// Initialize the variables.
	for (int i = 0; i < NLbf; i++) {
		thetaLbf[i] = 0.0;
	}
	// Initialize the parameters for the L-BFGS optimization.
	lbfgs_parameter_init(&param);
	//param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;

	//Start the L-BFGS optimization; this will invoke the callback functions
	//evaluate() and progress() when necessary.
	ret = lbfgs(NLbf, thetaLbf, &loglikeInnerLbf, evaluate1, progress1, NULL, &param);
	// Report the result. 
	cout << "L-BFGS optimization terminated with status code = " << ret << endl;
	//cout << "loglikeInnerLbf = " << loglikeInnerLbf << " " << "thetaLbf = " << thetaLbf[0] << endl;
	if (ret != 0) {
		if (thetaLbf[0] <= 0) {
			thetaLbf[0] = -50;
			loglikeInnerLbf = loglikeInner(thetaLbf[0]);
		}
		else {
			thetaLbf[0] = 10;
			loglikeInnerLbf = loglikeInner(thetaLbf[0]);
		}
	}

	lambda = exp(thetaLbf[0]);
	fn1 = loglikeInnerLbf;
	fn0 = loglikeInner(-DBL_MAX);
	lrt = 2 * (fn0 - fn1);
	parmfix = fixedInner(lambda);
	lod = lrt / 4.61;
	if (lrt < 0) { lrt0 = 0; }
	else { lrt0 = lrt; }
	boost::math::chi_squared mydist(1);
	pvalue = 1 - boost::math::cdf(mydist, lrt0);
	sigma2g = lambda * parmfix[2 * s];
	parameter[0] = lrt;
	parameter[1] = parmfix[0];
	parameter[2] = parmfix[1];
	parameter[3] = parmfix[2];
	parameter[4] = lambda;
	parameter[5] = sigma2g;
	parameter[6] = lod;
	parameter[7] = pvalue;
	lbfgs_free(thetaLbf);
	//parameter includes lrt,beta,stderr,sigma2,lambda,sigma2g,lod,pvalue
	return parameter;
}

double loglike(double theta)
{
	int N1 = delta.size();
	double xi, yHy, yHx, logdt2, tmp0, tmp, xHx, loglike1;

	xi = exp(theta);
	tmp0 = zz * xi + 1;
	tmp = xi * (1 / tmp0);
	yHy = yy1 - zy * tmp * zy;
	yHx = yx1 - zx * tmp * zy;
	xHx = xx1 - zx * tmp * zx;
	logdt2 = log(tmp0);
	loglike1 = -0.5 * logdt2 - 0.5 * (N1 - s) * log(yHy - yHx * (1 / xHx) * yHx) - 0.5 * log(xHx);
	return (-loglike1);
}

vector<double> fixed1(double xi)
{
	int N1 = delta.size();
	double sigma2, gamma, var, stderror, tmp0, tmp, tmp2, xHx, yHy, yHx, zHy, zHx, zHz, beta;
	vector<double> fixedres1(4);

	tmp0 = zz * xi + 1;
	tmp = xi * (1 / tmp0);
	yHy = yy1 - zy * tmp * zy;
	yHx = yx1 - zx * tmp * zy;
	xHx = xx1 - zx * tmp * zx;
	zHy = zy - zz * tmp * zy;
	zHx = zx - zx * tmp * zz;
	zHz = zz - zz * tmp * zz;
	beta = (1 / xHx) * yHx;
	tmp2 = (1 / xHx);
	sigma2 = (yHy - yHx * tmp2 * yHx) / (N1 - s);
	gamma = xi * zHy - xi * zHx * tmp2 * yHx;
	var = abs((xi - xi * zHz * xi) * sigma2);
	stderror = sqrt(var);
	fixedres1[0] = gamma;
	fixedres1[1] = stderror;
	fixedres1[2] = beta;
	fixedres1[3] = sigma2;
	//list(gamma,stderr,beta,sigma2)
	return fixedres1;
}

static lbfgsfloatval_t evaluate2(
	void* instance,
	const lbfgsfloatval_t* thetaLbf1,
	lbfgsfloatval_t* gLbf1,
	const int n,
	const lbfgsfloatval_t step
)
{
	lbfgsfloatval_t loglikeLbf = 0.0;
	for (int i = 0; i < n; i++) {
		if ((thetaLbf1[i] > -10) && (thetaLbf1[i] < 10)) {
			gLbf1[i] = boost::math::differentiation::finite_difference_derivative(loglike, thetaLbf1[i]);
			loglikeLbf = loglike(thetaLbf1[i]);
		}
	}
	return loglikeLbf;
}

static int progress2(
	void* instance,
	const lbfgsfloatval_t* thetaLbf1,
	const lbfgsfloatval_t* gLbf1,
	const lbfgsfloatval_t loglikeLbf,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step,
	int n,
	int k,
	int ls
)
{
	//cout << "Iteration:" << k << endl;
	//cout << "loglikeLbf = " << loglikeLbf << " " << "thetaLbf1 = " << thetaLbf1[0] << endl;
	//cout << "xnorm = " << xnorm << " " << "gnorm = " << gnorm << " " << "step = " << step << endl;
	//cout << endl;
	return 0;
}

vector<int> setdif(vector<int> sub, int ss, int jj)
{
	int temp_i = 0;
	vector<int> ii(ss - 1);
	for (int i = 0; i < ss; i++)
	{
		if (i != jj)
		{
			ii[temp_i] = i;
			temp_i++;
		}
	}
	return ii;
}

//NN is the rows of genotype, KK is the columns of genotype, MM is the columns of fixed matrix,z is genotype,x is fixed matrix
vector<double> PenalizedML(vector<vector<double>> z, vector<vector<double>> x, vector<vector<double>> y, int NN, int KK, int MM)
{
	double meanz = 0.0, v0 = 0.0, e0 = 0.0;
	int i, j, jj, v = 4;
	int Col_b = KK + MM;
	vector<double> b_full(Col_b);

	for (i = 0; i < NN; i++)
	{
		meanz = meanz + y[i][0];
	}
	meanz = meanz / NN;
	b_full[0] = meanz;
	for (i = 0; i < NN; i++)
	{
		e0 = e0 + 0.01 * pow((y[i][0] - b_full[0]), 2);
	}
	e0 = e0 / NN;
	v0 = e0;
	vector<double> u(Col_b);
	for (i = 1; i < Col_b; i++)
	{
		b_full[i] = 0.01 * sqrt(e0);
		u[i] = b_full[i] / (v + 1);
	}
	u[0] = b_full[0] / (v + 1);

	vector<double> b1(Col_b), e(Col_b), xx(NN * Col_b), wt(Col_b), nexus(NN);
	double esp = 1;
	int niter = 200, iter = 0;

	for (i = 0; i < Col_b; i++)
	{
		for (j = 0; j < NN; j++)
		{
			if (i < MM)
			{
				xx[j * Col_b + i] = x[j][i];
			}
			else
			{
				xx[j * Col_b + i] = z[j][(i - MM)];
			}
		}
	}
	for (i = 0; i < Col_b; i++)
	{
		for (j = 0; j < NN; j++)
		{
			wt[i] = wt[i] + pow(xx[j * Col_b + i], 2);
		}
		if (i >= MM)
		{
			e[i] = 0.0005 * e0;
		}
	}
	vector<double> r(NN);
	vector<int> ii(Col_b - 1), sub(Col_b);
	double rtemp = 0, v1;
	for (i = 0; i < Col_b; i++)
	{
		sub[i] = i;
	}

	while ((esp > 1e-8) && (iter < niter))
	{
		iter = iter + 1;
		//printf("%d\t%e\n",iter,esp);
		for (i = 0; i < Col_b; i++)
		{
			b1[i] = b_full[i];
		}
		v1 = v0;
		if (iter > 5)
		{
			v0 = 0;
			for (i = 0; i < NN; i++)
			{
				r[i] = 0;
				for (j = 0; j < Col_b; j++)
				{
					r[i] = r[i] + xx[j + i * Col_b] * b_full[j];
				}
				r[i] = y[i][0] - r[i];
				v0 = v0 + pow(r[i], 2) / NN;
			}
		}
		else
		{
			v0 = e0;
		}

		for (j = 0; j < MM; j++)
		{
			rtemp = 0;
			ii = setdif(sub, Col_b, j);
			for (i = 0; i < NN; i++)
			{
				r[i] = 0;
				for (jj = 0; jj < (Col_b - 1); jj++)
				{
					r[i] = r[i] + xx[ii[jj] + i * Col_b] * b_full[ii[jj]];
				}
				r[i] = y[i][0] - r[i];
				rtemp = rtemp + r[i] * xx[i * Col_b + j];
			}
			b_full[j] = rtemp / wt[j];
		}

		for (i = 0; i < NN; i++)
		{
			nexus[i] = 0;
			for (j = 0; j < Col_b; j++)
			{
				nexus[i] = nexus[i] + xx[j + i * Col_b] * b_full[j];
			}
		}

		for (j = MM; j < Col_b; j++)
		{
			if ((iter > 5) && (abs(b_full[j]) < 1e-4))
			{
				b_full[j] = 0;
			}
			else
			{
				rtemp = 0;
				for (i = 0; i < NN; i++)
				{
					nexus[i] = nexus[i] - xx[j + i * Col_b] * b_full[j];
					rtemp = rtemp + (y[i][0] - nexus[i]) * xx[j + i * Col_b];
				}
				b_full[j] = (rtemp + u[j] * v0 / e[j]) / (wt[j] + v0 / e[j]);
				for (i = 0; i < NN; i++)
				{
					nexus[i] = nexus[i] + xx[j + i * Col_b] * b_full[j];
				}
			}

			u[j] = b_full[j] / (v + 1);
			if ((iter > 5) && (e[j] < 1e-8))
			{
				e[j] = 1e-300;
			}
			else
			{
				e[j] = (pow((b_full[j] - u[j]), 2) + v * pow(u[j], 2)) / 2;
			}
		}

		esp = 0;
		for (i = 0; i < Col_b; i++)
		{
			esp = esp + pow((b1[i] - b_full[i]), 2) / (Col_b + 1);
		}
		esp = esp + pow((v1 - v0), 2) / (Col_b + 1);
	}

	return b_full;
}

double Normal_Sum(vector<vector<double>> y, vector<vector<double>> means, double var_v)
{
	int m = means.size();
	vector<double> pdf_value(m);
	double sum_log = 0.0;
	for (int i = 0; i < m; i++)
	{
		pdf_value[i] = (1 / sqrt(2 * 3.14159265358979323846 * var_v)) * exp(-pow((y[i][0] - means[i][0]), 2) / (2 * var_v));
		sum_log = sum_log + log(abs(pdf_value[i]));
	}

	return sum_log;
}

vector<double> LikelihoodTest(vector<vector<double>> fix, vector<vector<double>> genoPML, vector<vector<double>> pheno)
{
	int ns = genoPML.size();
	int nq = genoPML[0].size();
	int ncolfix = fix[0].size();
	vector<double> lod(nq);
	vector<vector<double>> ad(ns, vector<double>(ncolfix + nq));
	for (int i = 0; i < ns; i++) {
		for (int j = 0; j < (ncolfix + nq); j++) {
			if (j < ncolfix) {
				ad[i][j] = fix[i][j];
			}
			else {
				ad[i][j] = genoPML[i][j - ncolfix];
			}
		}
	}

	vector<vector<double>> adtad = MatrixMultiply(Transpose(ad), ad);
	vector<vector<double>> vecval = Eigvecval(adtad), tmpadtad = adtad;
	vector<vector<double>> bb, adtyn = MatrixMultiply(Transpose(ad), pheno);
	int coladtad = adtad.size(), countSmall = 0;
	double vv1 = 0.0, cc1;
	for (int i = 0; i < coladtad; i++) {
		if (vecval[coladtad][i] < 1e-6) {
			countSmall = countSmall + 1;
		}
	}
	if (countSmall > 0) {
		for (int i = 0; i < coladtad; i++) {
			tmpadtad[i][i] = tmpadtad[i][i] + 0.01;
		}
		bb = MatrixMultiply(MatrixInverse(tmpadtad), adtyn);
	}
	else {
		bb = MatrixMultiply(MatrixInverse(adtad), adtyn);
	}

	for (int i = 0; i < ns; i++) {
		vv1 = vv1 + pow((pheno[i][0] - MatrixMultiply(ad, bb)[i][0]), 2);
	}
	vv1 = vv1 / ns;
	cc1 = Normal_Sum(pheno, MatrixMultiply(ad, bb), vv1);

	vector<int> sub(coladtad), ij(coladtad - 1);
	for (int i = 0; i < coladtad; i++) {
		sub[i] = i;
	}

	int countSmall1; double vv0, cc0;
	vector<vector<double>> ad1(ns, vector<double>(coladtad - 1));
	vector<vector<double>> ad1tad1, vecval1, tmpad1tad1, bb1, ad1tyn;

	for (int hh = 0; hh < nq; hh++) {
		countSmall1 = 0;
		vv0 = 0.0;
		ij = setdif(sub, coladtad, (hh + ncolfix));
		for (int i = 0; i < ns; i++) {
			for (int j = 0; j < (coladtad - 1); j++) {
				ad1[i][j] = ad[i][ij[j]];
			}
		}
		ad1tad1 = MatrixMultiply(Transpose(ad1), ad1);
		vecval1 = Eigvecval(ad1tad1);
		tmpad1tad1 = ad1tad1;
		ad1tyn = MatrixMultiply(Transpose(ad1), pheno);
		for (int i = 0; i < (coladtad - 1); i++) {
			if (vecval1[coladtad - 1][i] < 1e-6) {
				countSmall1 = countSmall1 + 1;
			}
		}
		if (countSmall1 > 0) {
			for (int i = 0; i < coladtad; i++) {
				tmpad1tad1[i][i] = tmpad1tad1[i][i] + 0.01;
			}
			bb1 = MatrixMultiply(MatrixInverse(tmpad1tad1), ad1tyn);
		}
		else {
			bb1 = MatrixMultiply(MatrixInverse(ad1tad1), ad1tyn);
		}
		for (int i = 0; i < ns; i++) {
			vv0 = vv0 + pow((pheno[i][0] - MatrixMultiply(ad1, bb1)[i][0]), 2);
		}
		vv0 = vv0 / ns;
		cc0 = Normal_Sum(pheno, MatrixMultiply(ad1, bb1), vv0);
		lod[hh] = -2.0 * (cc0 - cc1) / (2.0 * log(10));
	}

	return lod;
}


int main(int argc, char **argv)
{
	clock_t t1, t2;
	// Input genotype, phenotype, genetics relatedness matrix and covariance files
	string filegene = argv[1];
	string filepheno = argv[2];
	string filekinship = argv[3];
	string filefix = argv[4];
	// Output the result of method, and provide running time
	string outResults = argv[5];
	string outTime = argv[6];
	// repeatTime is set by 1 for real data analysis, and it can be changed as repeat times for simulated study
	int repeatTime = 1;
	vector<vector<double>> fixmat = ReadFile(filefix);
	vector<vector<double>> geno = ReadFile(filegene);
	vector<vector<double>> pheno = ReadFile(filepheno);
	vector<vector<double>> kins = ReadFile(filekinship);

	t1 = clock();

	vector<vector<double>> genoTran = Transpose(geno);
	uint nrgeno = genoTran.size() - 2; //n <- nrow(gen)
	uint ncgeno = genoTran[0].size();  //m <- ncol(gen)
	vector<vector<double>> X(nrgeno, vector<double>(1));
	for (uint i = 0; i < nrgeno; i++) {    //x <- matrix(1,n,1)
		for (uint j = 0; j < 1; j++) {
			X[i][j] = 1;
		}
	}
	s = 1;   //s <- 1
	vector<vector<double>> eigvecval = Eigvecval(kins); //Each row corresponding to one eigenvector, the last row is eigenvalue
	vector<vector<double>> uu(nrgeno, vector<double>(nrgeno));
	delta = MatrixMultiply(uu, X);
	for (uint i = 0; i < nrgeno; i++) {
		delta[i][0] = eigvecval[nrgeno][i];
	}

	for (uint i = 0; i < nrgeno; i++)
	{
		for (uint j = 0; j < nrgeno; j++) {
			uu[i][j] = eigvecval[i][j];
		}
	}
	xu = MatrixMultiply(uu, X);
	vector<vector<double>> yy(nrgeno, vector<double>(1));
	vector<double> parammixed;//length is (2 * s + 6)
	vector<double> paramfix;
	vector<vector<double>> z(nrgeno, vector<double>(1));
	vector<vector<double>> zu;
	double lambda1, xi1, foutn1, foutn0;
	double gamma, stderror, beta, sigma2, lrt1, lrt2, p_lrt, wald, p_wald, sigma2g;

	int ret1;
	lbfgsfloatval_t loglikeLbf;
	lbfgsfloatval_t* thetaLbf1 = lbfgs_malloc(NLbf);
	lbfgs_parameter_t param1;

	vector<vector<double>> RlmmRes(ncgeno, vector<double>(4));
	int Psmallnum, tempPnum, nrgenoPen, ncgenoPen, ncfixPen, nonzero;
	vector<double> PMLres, Lodvalue;

	vector<vector<double>> fixmat1(nrgeno, vector<double>(1));
	int ncfixEff = fixmat[0].size();
	if (ncfixEff == 1) {
		fixmat1 = fixmat;
	}
	else {
		for (int i = 0; i < nrgeno; i++) {
			fixmat1[i][0] = fixmat[i][0];
		}
		vector<vector<double>> fixtfix = MatrixMultiply(Transpose(fixmat), fixmat);
		vector<vector<double>> fixty = MatrixMultiply(Transpose(fixmat), pheno);
		vector<vector<double>> bbfix, vecvalfix = Eigvecval(fixtfix), tmpfixtfix = fixtfix;
		int colfixtfix = fixtfix.size(), countSmallfix = 0;
		for (int i = 0; i < colfixtfix; i++) {
			if (vecvalfix[colfixtfix][i] < 1e-6) {
				countSmallfix = countSmallfix + 1;
			}
		}
		if (countSmallfix > 0) {
			for (int i = 0; i < colfixtfix; i++) {
				tmpfixtfix[i][i] = tmpfixtfix[i][i] + 0.01;
			}
			bbfix = MatrixMultiply(MatrixInverse(tmpfixtfix), fixty);
		}
		else {
			bbfix = MatrixMultiply(MatrixInverse(fixtfix), fixty);
		}
		for (int i = 1; i < colfixtfix; i++)
		{
			for (int j = 0; j < nrgeno; j++)
			{
				pheno[j][0] = pheno[j][0] - fixmat[j][i] * bbfix[i][0];
			}
		}
	}

	ofstream outputRes;
	outputRes.open(outResults, std::ios_base::app);
	outputRes << "Repeat,Number,Chr,Pos,Wald,Pvalue,Lod,PMLbeta,Rlmmbeta" << endl;

	//here ii is repeat time
	for (uint ii = 0; ii < repeatTime; ii++) {
		cout << ii << endl;
		yy1 = 0.0;
		yx1 = 0.0;
		xx1 = 0.0;
		h1 = xu;

		for (uint j = 0; j < nrgeno; j++) {
			yy[j][0] = pheno[ii * nrgeno + j][0];
			//cout << yy[j][0] << endl;
		}
		yu = MatrixMultiply(uu, yy);
		parammixed = mixed(X, yy, kins);
		//for (int j = 0; j < 8; j++) {
		//	cout << parammixed[j] << endl;
		//}
		lambda1 = parammixed[4];

		for (int i = 0; i < nrgeno; i++) {
			h1[i][0] = 1 / (lambda1 * delta[i][0] + 1);
			xx1 = xx1 + xu[i][0] * h1[i][0] * xu[i][0];
			yy1 = yy1 + yu[i][0] * h1[i][0] * yu[i][0];
			yx1 = yx1 + yu[i][0] * h1[i][0] * xu[i][0];
		}

		for (int pp = 0; pp < ncgeno; pp++) {
			for (uint i = 0; i < nrgeno; i++) {
				z[i][0] = genoTran[i + 2][pp];
			}

			zu = MatrixMultiply(uu, z);
			zx = 0.0;
			zy = 0.0;
			zz = 0.0;

			for (int i = 0; i < nrgeno; i++) {
				zy = zy + yu[i][0] * h1[i][0] * zu[i][0];
				zz = zz + zu[i][0] * h1[i][0] * zu[i][0];
				zx = zx + xu[i][0] * h1[i][0] * zu[i][0];
			}

			if (thetaLbf1 == NULL) {
				cerr << "ERROR: Failed to allocate a memory block for variables.\n" << endl;
			}
			for (int i = 0; i < NLbf; i++) {
				thetaLbf1[i] = 0.0;
			}
			lbfgs_parameter_init(&param1);
			ret1 = lbfgs(NLbf, thetaLbf1, &loglikeLbf, evaluate2, progress2, NULL, &param1);
			//cout << "L-BFGS optimization terminated with status code = " << ret1 << endl;
			//cout << "loglikeLbf = " << loglikeLbf << " " << "thetaLbf = " << thetaLbf1[0] << endl;

			if (ret1 != 0) {
				if (thetaLbf1[0] <= 0) {
					thetaLbf1[0] = -10;
					loglikeLbf = loglike(thetaLbf1[0]);
				}
				else {
					thetaLbf1[0] = 10;
					loglikeLbf = loglike(thetaLbf1[0]);
				}
			}

			//cout << thetaLbf1[0] << endl;
			xi1 = exp(thetaLbf1[0]);
			foutn1 = loglikeLbf;
			paramfix = fixed1(xi1);
			gamma = paramfix[0];
			stderror = paramfix[1];
			beta = paramfix[2];
			sigma2 = paramfix[3];
			lambda1 = xi1;
			sigma2g = lambda1 * sigma2;
			foutn0 = loglike(-DBL_MAX);
			lrt1 = 2 * (foutn0 - foutn1);
			boost::math::chi_squared mydist1(1);
			if (lrt1 < 0) { lrt2 = 0; }
			else { lrt2 = lrt1; }
			p_lrt = 1 - boost::math::cdf(mydist1, lrt2);
			wald = pow((gamma / stderror), 2);
			boost::math::chi_squared mydist2(1);
			p_wald = 1 - boost::math::cdf(mydist2, wald);

			RlmmRes[pp][0] = pp;
			RlmmRes[pp][1] = wald;
			RlmmRes[pp][2] = p_wald;
			RlmmRes[pp][3] = gamma;
		}

		Psmallnum = 0, tempPnum = 0;
		for (int i = 0; i < ncgeno; i++)
		{
			if (RlmmRes[i][2] < rlmmPvalue)
			{
				Psmallnum = Psmallnum + 1;
			}
		}
		vector<vector<double>> RetainRlmmres(Psmallnum, vector<double>(4));
		vector<vector<double>> genoPen(nrgeno, vector<double>(Psmallnum));
		for (int i = 0; i < ncgeno; i++)
		{
			if (RlmmRes[i][2] < rlmmPvalue)
			{
				RetainRlmmres[tempPnum][0] = RlmmRes[i][0];
				RetainRlmmres[tempPnum][1] = RlmmRes[i][1];
				RetainRlmmres[tempPnum][2] = RlmmRes[i][2];
				RetainRlmmres[tempPnum][3] = RlmmRes[i][3];
				tempPnum = tempPnum + 1;
			}
		}

		for (int i = 0; i < nrgeno; i++) {
			for (int j = 0; j < Psmallnum; j++) {
				genoPen[i][j] = genoTran[i + 2][int(RetainRlmmres[j][0])];
			}
		}

		nrgenoPen = genoPen.size();
		ncgenoPen = genoPen[0].size();
		ncfixPen = fixmat1[0].size();
		nonzero = 0;

		PMLres = PenalizedML(genoPen, fixmat1, yy, nrgenoPen, ncgenoPen, ncfixPen);

		for (int i = 0; i < (ncgenoPen + ncfixPen); i++)
		{
			if (PMLres[i] != 0) {
				nonzero = nonzero + 1;
			}
		}
		int jj0 = 0, jj1 = 0;
		vector<int> IndexPMLres(nonzero);
		for (int i = 0; i < (ncgenoPen + ncfixPen); i++)
		{
			if (PMLres[i] != 0) {
				IndexPMLres[jj0] = i;
				jj0 = jj0 + 1;
				if (i >= ncfixPen) {
					jj1 = jj1 + 1;
				}
			}
		}

		if (jj1 == 0) {
			cerr << "There is no SNP effect larger than 1e-4!" << endl;
		}

		vector<vector<double>> ChooseGen(nrgenoPen, vector<double>(jj1));
		vector<vector<double>> FinalRes(jj1, vector<double>(9));
		int finalIndex, tmpMIndex;
		for (int i = 0; i < nrgenoPen; i++) {
			for (int j = 0; j < jj1; j++) {
				ChooseGen[i][j] = genoPen[i][(IndexPMLres[j + ncfixPen] - ncfixPen)];
			}
		}
		Lodvalue = LikelihoodTest(fixmat1, ChooseGen, yy);

		for (int i = 0; i < jj1; i++) {
			tmpMIndex = (IndexPMLres[i + ncfixPen] - ncfixPen);
			finalIndex = RetainRlmmres[(IndexPMLres[i + ncfixPen] - ncfixPen)][0];
			FinalRes[i][0] = ii + 1;
			FinalRes[i][1] = RetainRlmmres[tmpMIndex][0] + 1;
			FinalRes[i][2] = genoTran[0][finalIndex];
			FinalRes[i][3] = genoTran[1][finalIndex];
			FinalRes[i][4] = RetainRlmmres[tmpMIndex][1];
			FinalRes[i][5] = RetainRlmmres[tmpMIndex][2];
			FinalRes[i][6] = Lodvalue[i];
			FinalRes[i][7] = PMLres[tmpMIndex + ncfixPen];
			FinalRes[i][8] = RetainRlmmres[tmpMIndex][3];
		}

		for (int i = 0; i < jj1; i++)
		{
			for (int j = 0; j < 3; j++) {
				outputRes << FinalRes[i][j] << ",";
			}
			outputRes << std::to_string(FinalRes[i][3]) << ",";
			for (int j = 4; j < 9; j++) {
				outputRes << FinalRes[i][j] << ",";
			}
			outputRes << endl;
		}

	}
	lbfgs_free(thetaLbf1);

	outputRes.close();


	ofstream outputtime;
	outputtime.open(outTime, std::ios_base::app);
	outputtime << "Running Time:" << endl;

	t2 = clock();
	double ttime;
	ttime = double(t2 - t1) / CLOCKS_PER_SEC;
	cout << endl;
	cout << ttime << endl;

	outputtime << ttime << endl;
	outputtime.close();


	return 0;
}