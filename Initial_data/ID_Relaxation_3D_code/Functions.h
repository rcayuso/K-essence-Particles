#ifndef included_FunctionsXD
#define included_FunctionsXD

#include "SAMRAI/tbox/Utilities.h"
#include <string>
#include <math.h>
#include <vector>
#include "hdf5.h"
#include "Commons.h"


#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define SIGN(X) (((X) > 0) - ((X) < 0))
#define greaterEq(a,b) ((fabs((a) - (b))/1.0E-15 > 10 ? false : (floor(fabs((a) - (b))/1.0E-15) < 1)) || (b)<(a))
#define lessEq(a,b) ((fabs((a) - (b))/1.0E-15 > 10 ? false: (floor(fabs((a) - (b))/1.0E-15) < 1)) || (a)<(b))
#define equalsEq(a,b) ((fabs((a) - (b))/1.0E-15 > 10 ? false: (floor(fabs((a) - (b))/1.0E-15) < 1)))
#define reducePrecision(x, p) (floor(((x) * pow(10, (p)) + 0.5)) / pow(10, (p)))

using namespace external;

/*
 * Calculates coefficients for quintic lagrangian interpolation.
 *    coefs;      coefficients to obtain
 *    coord:      point in which the interpolation must be calculated
 *    position:   surrounding points for interpolation
 */
inline void calculateCoefficientsMapFile(double* coefs, double coord, double* position) {

    for (int i = 0; i < 6; i++) {
        coefs[i] = 0;
    }
    //Search for a perfect fit

    for (int i = 0; i < 6; i++) {
        if (position[i] == coord) {
            coefs[i] = 1;
            return;
        }
    }

    double x1 = position[0];
    double x2 = position[1];
    double x3 = position[2];
    double x4 = position[3];
    double x5 = position[4];
    double x6 = position[5];

    coefs[0] = (coord-x2)*(coord-x3)*(coord-x4)*(coord-x5)*(coord-x6) / ((x1-x2)*(x1-x3)*(x1-x4)*(x1-x5)*(x1-x6));
    coefs[1] = (coord-x1)*(coord-x3)*(coord-x4)*(coord-x5)*(coord-x6) / ((x2-x1)*(x2-x3)*(x2-x4)*(x2-x5)*(x2-x6));
    coefs[2] = (coord-x1)*(coord-x2)*(coord-x4)*(coord-x5)*(coord-x6) / ((x3-x1)*(x3-x2)*(x3-x4)*(x3-x5)*(x3-x6));
    coefs[3] = (coord-x1)*(coord-x2)*(coord-x3)*(coord-x5)*(coord-x6) / ((x4-x1)*(x4-x2)*(x4-x3)*(x4-x5)*(x4-x6));
    coefs[4] = (coord-x1)*(coord-x2)*(coord-x3)*(coord-x4)*(coord-x6) / ((x5-x1)*(x5-x2)*(x5-x3)*(x5-x4)*(x5-x6));
    coefs[5] = (coord-x1)*(coord-x2)*(coord-x3)*(coord-x4)*(coord-x5) / ((x6-x1)*(x6-x2)*(x6-x3)*(x6-x4)*(x6-x5));
}

/*
 * Reads fields from a table interpolating in given coordinates.
 * Uses quintic lagrangian interpolation.
 *    nCoords:            number of coordinates of table
 *    nOutFields:         number of fields to read
 *    coord0, ...:        coordinate values to search in the table
 *    data0, ...:         output variables to store interpolated fields
 *    data0Index, ...:    index of fields inside the table
 *    coord0Data, ...:    variables storing all values of table coordinates
 *    coord0Dx, ...:      spacing of table coordinates
 *    coord0Size, ...:    size of table coordinates
 *    varData, ...:       table data
 *    nVars, ...:         number of variables in the table
 */
inline void readFromFileQuintic(int nCoords, int nOutFields, double coord0, double &data0, int data0Index, double *coord0Data, double coord0Dx, int coord0Size, double *varData, int nVars) {
	int i0  = int((coord0 - coord0Data[0])/coord0Dx);

	if (i0 >= coord0Size || i0 < 0) {
		TBOX_ERROR("Mapping File: the input file does not have a value for coordinate "<<coord0);
	}

	int i0b = std::min(std::max(i0-2,0),coord0Size-6);
	double coefs0[6];
	calculateCoefficientsMapFile(coefs0, coord0, &coord0Data[i0b]);
	data0 = 0;
	for (int i = 0; i < 6; i++) {
		data0 = data0 + coefs0[i] *  varData[data0Index + nVars * (i0b + i)];
	}
}


#define vector(v, i, j, k) (v)[i+ilast*(j+jlast*(k))]

#define POLINT_MACRO_LINEAR_1(y1, y2, i_ext, s_ext, dx, simPlat_dt, ilast, jlast) (-i_ext*y1 + i_ext*y2 + s_ext*y2)
#define POLINT_MACRO_LINEAR_2(y1, y2, i_ext, s_ext, dx, simPlat_dt, ilast, jlast) (-i_ext*y1 + i_ext*y2 - s_ext*y1)
#define POLINT_MACRO_QUADRATIC_1(y1,y2,y3, i_ext, s_ext, dx, simPlat_dt, ilast, jlast) (0.5*(i_ext*i_ext*y1-2*i_ext*i_ext*y2+i_ext*i_ext*y3+i_ext*s_ext*y1-4*i_ext*s_ext*y2+3*i_ext*s_ext*y3+2*s_ext*s_ext*y3)/(s_ext*s_ext))
#define POLINT_MACRO_QUADRATIC_2(y1,y2,y3, i_ext, s_ext, dx, simPlat_dt, ilast, jlast) (0.5*(i_ext*i_ext*y1-2*i_ext*i_ext*y2+i_ext*i_ext*y3+3*i_ext*s_ext*y1-4*i_ext*s_ext*y2+i_ext*s_ext*y3+2*s_ext*s_ext*y1)/(s_ext*s_ext))
#define POLINT_MACRO_CUBIC_1(y1,y2,y3, y4, i_ext, s_ext, dx, simPlat_dt, ilast, jlast) (-(1.0/6.0)*(i_ext*i_ext*i_ext*y1-3*i_ext*i_ext*i_ext*y2+3*i_ext*i_ext*i_ext*y3-i_ext*i_ext*i_ext*y4+3*i_ext*i_ext*s_ext*y1-12*i_ext*i_ext*s_ext*y2+15*i_ext*i_ext*s_ext*y3-6*i_ext*i_ext*s_ext*y4+2*i_ext*s_ext*s_ext*y1-9*i_ext*s_ext*s_ext*y2+18*i_ext*s_ext*s_ext*y3-11*i_ext*s_ext*s_ext*y4-6*s_ext*s_ext*s_ext*y4)/(s_ext*s_ext*s_ext))
#define POLINT_MACRO_CUBIC_2(y1,y2,y3, y4, i_ext, s_ext, dx, simPlat_dt, ilast, jlast) ((1.0/6.0)*(i_ext*i_ext*i_ext*y1-3*i_ext*i_ext*i_ext*y2+3*i_ext*i_ext*i_ext*y3-i_ext*i_ext*i_ext*y4+6*i_ext*i_ext*s_ext*y1-15*i_ext*i_ext*s_ext*y2+12*i_ext*i_ext*s_ext*y3-3*i_ext*i_ext*s_ext*y4+11*i_ext*s_ext*s_ext*y1-18*i_ext*s_ext*s_ext*y2+9*i_ext*s_ext*s_ext*y3-2*i_ext*s_ext*s_ext*y4+6*s_ext*s_ext*s_ext*y1)/(s_ext*s_ext*s_ext))

#define D1CDO4_i(paru, pari, parj, park, dx, simPlat_dt, ilast, jlast) ((((-vector(paru, (pari) + 2, (parj), (park))) + 8.0 * vector(paru, (pari) + 1, (parj), (park))) + ((-8.0 * vector(paru, (pari) - 1, (parj), (park))) + vector(paru, (pari) - 2, (parj), (park)))) / (12.0 * dx[0]))

#define D1CDO4_j(paru, pari, parj, park, dx, simPlat_dt, ilast, jlast) ((((-vector(paru, (pari), (parj) + 2, (park))) + 8.0 * vector(paru, (pari), (parj) + 1, (park))) + ((-8.0 * vector(paru, (pari), (parj) - 1, (park))) + vector(paru, (pari), (parj) - 2, (park)))) / (12.0 * dx[1]))

#define D1CDO4_k(paru, pari, parj, park, dx, simPlat_dt, ilast, jlast) ((((-vector(paru, (pari), (parj), (park) + 2)) + 8.0 * vector(paru, (pari), (parj), (park) + 1)) + ((-8.0 * vector(paru, (pari), (parj), (park) - 1)) + vector(paru, (pari), (parj), (park) - 2))) / (12.0 * dx[2]))

#define meshDissipation_i(paru, pari, parj, park, dx, simPlat_dt, ilast, jlast) ((vector(paru, (pari) + 3, (parj), (park)) + (-6.0 * vector(paru, (pari) + 2, (parj), (park))) + 15.0 * vector(paru, (pari) + 1, (parj), (park)) + (-20.0 * vector(paru, (pari), (parj), (park))) + 15.0 * vector(paru, (pari) - 1, (parj), (park)) + (-6.0 * vector(paru, (pari) - 2, (parj), (park))) + vector(paru, (pari) - 3, (parj), (park))) / (64.0 * dx[0]))

#define meshDissipation_j(paru, pari, parj, park, dx, simPlat_dt, ilast, jlast) ((vector(paru, (pari), (parj) + 3, (park)) + (-6.0 * vector(paru, (pari), (parj) + 2, (park))) + 15.0 * vector(paru, (pari), (parj) + 1, (park)) + (-20.0 * vector(paru, (pari), (parj), (park))) + 15.0 * vector(paru, (pari), (parj) - 1, (park)) + (-6.0 * vector(paru, (pari), (parj) - 2, (park))) + vector(paru, (pari), (parj) - 3, (park))) / (64.0 * dx[1]))

#define meshDissipation_k(paru, pari, parj, park, dx, simPlat_dt, ilast, jlast) ((vector(paru, (pari), (parj), (park) + 3) + (-6.0 * vector(paru, (pari), (parj), (park) + 2)) + 15.0 * vector(paru, (pari), (parj), (park) + 1) + (-20.0 * vector(paru, (pari), (parj), (park))) + 15.0 * vector(paru, (pari), (parj), (park) - 1) + (-6.0 * vector(paru, (pari), (parj), (park) - 2)) + vector(paru, (pari), (parj), (park) - 3)) / (64.0 * dx[2]))

#define RK4P1_(parRHS, parQn, dx, simPlat_dt, ilast, jlast) ((parQn) + (simPlat_dt * (parRHS)) / 2.0)

#define RK4P2_(parRHS, parQn, dx, simPlat_dt, ilast, jlast) ((parQn) + (simPlat_dt * (parRHS)) / 2.0)

#define RK4P3_(parRHS, parQn, dx, simPlat_dt, ilast, jlast) ((parQn) + simPlat_dt * (parRHS))

#define RK4P4_(parRHS, parQn, parQK1, parQK2, parQK3, dx, simPlat_dt, ilast, jlast) ((-(parQn) / 3.0) + (parQK1) / 3.0 + 2.0 * (parQK2) / 3.0 + (parQK3) / 3.0 + (simPlat_dt * (parRHS)) / 6.0)

#define Unit_MCD(i, j, k) ((((int)i) % 10 == 0 && ((int)j) % 10 == 0 && ((int)k) % 10 == 0) ? 10 : (((int)i) % 9 == 0 && ((int)j) % 9 == 0 && ((int)k) % 9 == 0) ? 9 : (((int)i) % 8 == 0 && ((int)j) % 8 == 0 && ((int)k) % 8 == 0) ? 8 : (((int)i) % 7 == 0 && ((int)j) % 7 == 0 && ((int)k) % 7 == 0) ? 7 : (((int)i) % 6 == 0 && ((int)j) % 6 == 0 && ((int)k) % 6 == 0) ? 6 : (((int)i) % 5 == 0 && ((int)j) % 5 == 0 && ((int)k) % 5 == 0) ? 5 : (((int)i) % 4 == 0 && ((int)j) % 4 == 0 && ((int)k) % 4 == 0) ? 4 : (((int)i) % 3 == 0 && ((int)j) % 3 == 0 && ((int)k) % 3 == 0) ? 3 : (((int)i) % 2 == 0 && ((int)j) % 2 == 0 && ((int)k) % 2 == 0) ? 2 : 1)

inline void extrapolate_field(double* pard_i_f, int pari, double* pard_j_f, int parj, double* pard_k_f, int park, double* parto_f, double* parFOV, const double* dx, const int ilast, const int jlast) {
	double sign_i_ext, sign_j_ext, sign_k_ext, disp_i_j, disp_i_k, disp_j_i, disp_j_k, disp_k_i, disp_k_j, y1, y2, y3, to_i_f, to_j_f, to_k_f, mod_d, term_i, term_j, term_k;

	sign_i_ext = SIGN(vector(pard_i_f, pari, parj, park));
	sign_j_ext = SIGN(vector(pard_j_f, pari, parj, park));
	sign_k_ext = SIGN(vector(pard_k_f, pari, parj, park));
	disp_i_j = 0.0;
	disp_i_k = 0.0;
	disp_j_i = 0.0;
	disp_j_k = 0.0;
	disp_k_i = 0.0;
	disp_k_j = 0.0;
	if (!equalsEq(vector(pard_i_f, pari, parj, park), 0.0) && equalsEq(vector(parFOV, int(pari - vector(pard_i_f, pari, parj, park)), parj, park), 0.0)) {
		disp_j_i = vector(pard_j_f, pari, parj, park);
		disp_k_i = vector(pard_k_f, pari, parj, park);
	}
	if (!equalsEq(vector(pard_j_f, pari, parj, park), 0.0) && equalsEq(vector(parFOV, pari, int(parj - vector(pard_j_f, pari, parj, park)), park), 0.0)) {
		disp_i_j = vector(pard_i_f, pari, parj, park);
		disp_k_j = vector(pard_k_f, pari, parj, park);
	}
	if (!equalsEq(vector(pard_k_f, pari, parj, park), 0.0) && equalsEq(vector(parFOV, pari, parj, int(park - vector(pard_k_f, pari, parj, park))), 0.0)) {
		disp_i_k = vector(pard_i_f, pari, parj, park);
		disp_j_k = vector(pard_j_f, pari, parj, park);
	}
	if (vector(pard_i_f, pari, parj, park) > 0.0) {
		y1 = vector(parto_f, int(pari - (vector(pard_i_f, pari, parj, park) + 2 * sign_i_ext)), int(parj - disp_j_i), int(park - disp_k_i));
		y2 = vector(parto_f, int(pari - (vector(pard_i_f, pari, parj, park) + sign_i_ext)), int(parj - disp_j_i), int(park - disp_k_i));
		y3 = vector(parto_f, int(pari - vector(pard_i_f, pari, parj, park)), int(parj - disp_j_i), int(park - disp_k_i));
		to_i_f = POLINT_MACRO_QUADRATIC_1(y1, y2, y3, vector(pard_i_f, pari, parj, park), sign_i_ext, dx, simPlat_dt, ilast, jlast);
	}
	if (vector(pard_i_f, pari, parj, park) < 0.0) {
		y3 = vector(parto_f, int(pari - (vector(pard_i_f, pari, parj, park) + 2 * sign_i_ext)), int(parj - disp_j_i), int(park - disp_k_i));
		y2 = vector(parto_f, int(pari - (vector(pard_i_f, pari, parj, park) + sign_i_ext)), int(parj - disp_j_i), int(park - disp_k_i));
		y1 = vector(parto_f, int(pari - vector(pard_i_f, pari, parj, park)), int(parj - disp_j_i), int(park - disp_k_i));
		to_i_f = POLINT_MACRO_QUADRATIC_2(y1, y2, y3, vector(pard_i_f, pari, parj, park), sign_i_ext, dx, simPlat_dt, ilast, jlast);
	}
	if (vector(pard_j_f, pari, parj, park) > 0.0) {
		y1 = vector(parto_f, int(pari - disp_i_j), int(parj - (vector(pard_j_f, pari, parj, park) + 2 * sign_j_ext)), int(park - disp_k_j));
		y2 = vector(parto_f, int(pari - disp_i_j), int(parj - (vector(pard_j_f, pari, parj, park) + sign_j_ext)), int(park - disp_k_j));
		y3 = vector(parto_f, int(pari - disp_i_j), int(parj - vector(pard_j_f, pari, parj, park)), int(park - disp_k_j));
		to_j_f = POLINT_MACRO_QUADRATIC_1(y1, y2, y3, vector(pard_j_f, pari, parj, park), sign_j_ext, dx, simPlat_dt, ilast, jlast);
	}
	if (vector(pard_j_f, pari, parj, park) < 0.0) {
		y3 = vector(parto_f, int(pari - disp_i_j), int(parj - (vector(pard_j_f, pari, parj, park) + 2 * sign_j_ext)), int(park - disp_k_j));
		y2 = vector(parto_f, int(pari - disp_i_j), int(parj - (vector(pard_j_f, pari, parj, park) + sign_j_ext)), int(park - disp_k_j));
		y1 = vector(parto_f, int(pari - disp_i_j), int(parj - vector(pard_j_f, pari, parj, park)), int(park - disp_k_j));
		to_j_f = POLINT_MACRO_QUADRATIC_2(y1, y2, y3, vector(pard_j_f, pari, parj, park), sign_j_ext, dx, simPlat_dt, ilast, jlast);
	}
	if (vector(pard_k_f, pari, parj, park) > 0.0) {
		y1 = vector(parto_f, int(pari - disp_i_k), int(parj - disp_j_k), int(park - (vector(pard_k_f, pari, parj, park) + 2 * sign_k_ext)));
		y2 = vector(parto_f, int(pari - disp_i_k), int(parj - disp_j_k), int(park - (vector(pard_k_f, pari, parj, park) + sign_k_ext)));
		y3 = vector(parto_f, int(pari - disp_i_k), int(parj - disp_j_k), int(park - vector(pard_k_f, pari, parj, park)));
		to_k_f = POLINT_MACRO_QUADRATIC_1(y1, y2, y3, vector(pard_k_f, pari, parj, park), sign_k_ext, dx, simPlat_dt, ilast, jlast);
	}
	if (vector(pard_k_f, pari, parj, park) < 0.0) {
		y3 = vector(parto_f, int(pari - disp_i_k), int(parj - disp_j_k), int(park - (vector(pard_k_f, pari, parj, park) + 2 * sign_k_ext)));
		y2 = vector(parto_f, int(pari - disp_i_k), int(parj - disp_j_k), int(park - (vector(pard_k_f, pari, parj, park) + sign_k_ext)));
		y1 = vector(parto_f, int(pari - disp_i_k), int(parj - disp_j_k), int(park - vector(pard_k_f, pari, parj, park)));
		to_k_f = POLINT_MACRO_QUADRATIC_2(y1, y2, y3, vector(pard_k_f, pari, parj, park), sign_k_ext, dx, simPlat_dt, ilast, jlast);
	}
	mod_d = sqrt(vector(pard_i_f, pari, parj, park) * vector(pard_i_f, pari, parj, park) + vector(pard_j_f, pari, parj, park) * vector(pard_j_f, pari, parj, park) + vector(pard_k_f, pari, parj, park) * vector(pard_k_f, pari, parj, park));
	term_i = 0.0;
	if (!equalsEq(vector(pard_i_f, pari, parj, park), 0.0)) {
		term_i = (to_i_f - vector(parto_f, int(pari - vector(pard_i_f, pari, parj, park)), int(parj - disp_j_i), int(park - disp_k_i))) / dx[0];
	}
	term_j = 0.0;
	if (!equalsEq(vector(pard_j_f, pari, parj, park), 0.0)) {
		term_j = (to_j_f - vector(parto_f, int(pari - disp_i_j), int(parj - vector(pard_j_f, pari, parj, park)), int(park - disp_k_j))) / dx[1];
	}
	term_k = 0.0;
	if (!equalsEq(vector(pard_k_f, pari, parj, park), 0.0)) {
		term_k = (to_k_f - vector(parto_f, int(pari - disp_i_k), int(parj - disp_j_k), int(park - vector(pard_k_f, pari, parj, park)))) / dx[2];
	}
	vector(parto_f, pari, parj, park) = vector(parto_f, int(pari - vector(pard_i_f, pari, parj, park)), int(parj - vector(pard_j_f, pari, parj, park)), int(park - vector(pard_k_f, pari, parj, park))) + (term_i + term_j + term_k) * sqrt((vector(pard_i_f, pari, parj, park) * dx[0]) * (vector(pard_i_f, pari, parj, park) * dx[0]) + (vector(pard_j_f, pari, parj, park) * dx[1]) * (vector(pard_j_f, pari, parj, park) * dx[1]) + (vector(pard_k_f, pari, parj, park) * dx[2]) * (vector(pard_k_f, pari, parj, park) * dx[2])) / mod_d;

};

void readTable(std::string inputFile, double **coord0, int &coord0Size, double &coord0Dx, double** data, int& nVars);
void readTable(std::string inputFile, double **coord0, int &coord0Size, double &coord0Dx, double** data, int& nVars);




#endif
