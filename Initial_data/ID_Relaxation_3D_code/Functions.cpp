#include "Functions.h"
#include <string>


#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define SIGN(X) (((X) > 0) - ((X) < 0))

using namespace external;

#define vector(v, i, j, k) (v)[i+ilast*(j+jlast*(k))]

/*
 * Reads a table in hdf5 file and stores all relevant data.
 * Only inputFile is an input parameter. The others are filled in this routine.
 *    inputFile:         Name of the table
 *    coord0, ...:       variable storing values of table dimensions
 *    coord0Size, ...:   size of table dimensions
 *    coord0Dx, ...:     spacing of table dimensions
 *    data:              variable data from the table
 *    nVars:             number of variables in the table
 */
void readTable(std::string inputFile, double **coord0, int &coord0Size, double &coord0Dx, double** data, int& nVars) {
	std::cout<<"*******************************"<<std::endl;
	std::cout<<"Reading table file: "<<inputFile<<std::endl;
	std::cout<<"*******************************"<<std::endl;

	hid_t file;
	file = H5Fopen(inputFile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

	// Use these two defines to easily read in a lot of variables in the same way
	// The first reads in one variable of a given type completely
	#define READ_EOS_HDF5(NAME,VAR,TYPE,MEM)                                      \
	do {                                                                        \
		hid_t dataset;                                                            \
		dataset = H5Dopen(file, NAME.c_str(), H5P_DEFAULT);                                \
		H5Dread(dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR);       \
		H5Dclose(dataset);                                            \
	} while (0)
	// The second reads a given variable into a hyperslab of the alltables_temp array
	#define READ_EOSTABLE_HDF5(NAME,OFF)                                     \
	do {                                                                   \
		hsize_t offset[2]     = {static_cast<hsize_t>(OFF),0};                                     \
		H5Sselect_hyperslab(mem3, H5S_SELECT_SET, offset, NULL, var3, NULL); \
		READ_EOS_HDF5(NAME,data_temp,H5T_NATIVE_DOUBLE,mem3);           \
	} while (0)

	// Read size of tables
	hid_t dataset;
	dataset = H5Dopen(file, "nvars", H5P_DEFAULT);
	H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nVars);
	H5Dclose(dataset);

	int rank;
	hid_t dataspace;
	std::string coordName;
	hsize_t* dims;
	// Read table coordinates
	dataset = H5Dopen(file, "coord0", H5P_DEFAULT);
	dataspace = H5Dget_space(dataset);
	rank = H5Sget_simple_extent_ndims(dataspace);
	dims = new hsize_t[rank];
	H5Sget_simple_extent_dims(dataspace, dims, dims);
	*coord0 = new double[dims[0]];
	coord0Size = dims[0];
	coordName = "coord0";
	READ_EOS_HDF5(coordName, *coord0, H5T_NATIVE_DOUBLE, H5S_ALL);
	coord0Dx = ((*coord0)[coord0Size - 1] - (*coord0)[0])/(coord0Size-1);

	// Allocate memory for table
	double* data_temp;
	if (!(data_temp = (double*)malloc(coord0Size  * nVars * sizeof(double)))) {
		TBOX_ERROR("Cannot allocate memory for input table "<<inputFile);
	}

	// Prepare HDF5 to read hyperslabs into data
	hsize_t table_dims[2] = {static_cast<hsize_t>(nVars), static_cast<hsize_t>((hsize_t)coord0Size )};
	hsize_t var3[2]       = { 1, (hsize_t)coord0Size };
	hid_t mem3 =  H5Screate_simple(2, table_dims, NULL);
	// Read data
	for (int i = 0; i < nVars; i++) {
		std::string varName = "var" + std::to_string(i);
		READ_EOSTABLE_HDF5(varName,  i);
	}
	H5Fclose(file);

	// change ordering of alltables array so that
	// the table kind is the fastest changing index
	if (!(*data = (double*)malloc(coord0Size  * nVars * sizeof(double)))) {
		TBOX_ERROR("Cannot allocate memory for input table "<<inputFile);
	}
	for(int iv = 0;iv<nVars;iv++) {
		for(int i0 = 0; i0<coord0Size; i0++) {
			int indold = (i0 + coord0Size * iv);
			int indnew = iv + nVars * i0;
			(*data)[indnew] = data_temp[indold];
		}
	}
	// free memory of temporary array
	free(data_temp);
}

