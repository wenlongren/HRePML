# HRePML: A fast C++ program to perform GWAS with hybrid of restricted and penalized maximum likelihood method

# Linux Version:
(Using Ubuntu 20.04 LTS system as example.)

**Installation and Configure**
1.	Install blas and lapack\
sudo apt-get install libblas-dev liblapack-dev

2.	Install boost library\
sudo apt-get install libboost-all-dev\
or access website https://www.boost.org/ and then download and install

3.	Install libLBFGS\
Download source code from http://www.chokkan.org/software/liblbfgs/ and then decompression, or directly download from our folder ../thirdParty/liblbfgs-1.10.tar.gz\
$ cd liblbfgs-1.10\
$ ./autogen.sh\
$ ./configure\
$ make\
$ make install\
$ export LD_LIBRARY_PATH= /path/liblbfgs-1.10/lib/.libs:$LD_LIBRARY_PATH

**Run HRePML**

1.	g++ -I /path/liblbfgs-1.10/include HRePML-Linux.cpp -llapack -lblas -llbfgs -o output\
For Intel CPU users, blas and lapack libraries can be linked to optimized math routine, that is Math Kernal Library (MKL). If MKL is installed, the following code are recommended:\
g++ -I /path/liblbfgs-1.10/include HRePML-Linux.cpp -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -llbfgs -o output

2.	./output genofile phenofile kinshipfile covariatesfile resultfile timefile\
Note: (1). only support **.csv** file format at present; (2). the first column of covariatesfile is **1** column vector



# Windows Version:
(Using Windows 10 system and Visual Studio 2019 as example.)

**Installation and Configure**

1.	Install clapack\
Refer and download clapack package from http://icl.cs.utk.edu/lapack-for-windows/clapack/, clapack-3.2.1 has been verified by us and it is no problem.

2.	Install boost library\
Download and decompress boost_1_73_0.zip file from https://www.boost.org/users/history/version_1_73_0.html

3.	Install libLBFGS\
Download source code from http://www.chokkan.org/software/liblbfgs/ and then decompression, or directly download from our folder ../thirdParty/liblbfgs-1.10.tar.gz\
Find /path/liblbfgs-1.10/lbfgs.sln file, open this file with Visual Studio 2019 and then build file, note whether debug or release mode.\
In windows system, there are several things to take care about. And especially focus on configure .header pathway and .lib pathway in Visual Studio 2019. A small neglect may cause the operation to fail. 

Line 721-727: The pathway and name of variables “filegene, filepheno, filekinship, filefix, outResults and outTime” can be changed flexibly by users.

**Please refer:**\
Ren W, Liang Z, He S, Xiao J. Hybrid of Restricted and Penalized Maximum Likelihood Method for Efficient Genome-Wide Association Study. Genes. 2020; 11(11):1286.
