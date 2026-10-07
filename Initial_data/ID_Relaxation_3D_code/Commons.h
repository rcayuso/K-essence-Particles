#ifndef included_CommonsXD
#define included_CommonsXD

#ifdef EXTERNAL_EOS
#include "reprimand/con2prim_imhd.h"
#include "reprimand/eos_idealgas.h"
#include "reprimand/eos_hybrid.h"
#include "reprimand/eos_barotropic.h"
#include "reprimand/eos_barotr_file.h"
#include "reprimand/eos_thermal_file.h"
#endif

//Simulation constants 
#define vector(v, i, j, k) (v)[i+ilast*(j+jlast*(k))]
#define vectorCell(v, i, j, k) (v)[i+(ilast - 1)*(j+(jlast - 1)*(k))]
#define vectorT(v, i, j, k) (v)[i+itlast*(j+jtlast*(k))]
#define indexiOf(i) (i - (int)(i/((boxlast(0)-boxfirst(0)+1)*(boxlast(1)-boxfirst(1)+1)))*((boxlast(0)-boxfirst(0)+1)*(boxlast(1)-boxfirst(1)+1)) - (int)((i - (int)(i/((boxlast(0)-boxfirst(0)+1)*(boxlast(1)-boxfirst(1)+1)))*((boxlast(0)-boxfirst(0)+1)*(boxlast(1)-boxfirst(1)+1)))/(boxlast(0)-boxfirst(0)+1))*(boxlast(0)-boxfirst(0)+1))
#define indexjOf(i) ((int)(i - (int)((i/((boxlast(0)-boxfirst(0)+1)*(boxlast(1)-boxfirst(1)+1)))*((boxlast(0)-boxfirst(0)+1)*(boxlast(1)-boxfirst(1)+1)))/(boxlast(0)-boxfirst(0)+1)))
#define indexkOf(i) ((int)(i/((boxlast(0)-boxfirst(0)+1)*(boxlast(1)-boxfirst(1)+1))))
#define xcoord(i) (((d_grid_geometry->getXLower()[0] + ((i + boxfirst(0)) - d_ghost_width) * dx[0])))
#define ycoord(j) (((d_grid_geometry->getXLower()[1] + ((j + boxfirst(1)) - d_ghost_width) * dx[1])))
#define zcoord(k) (((d_grid_geometry->getXLower()[2] + ((k + boxfirst(2)) - d_ghost_width) * dx[2])))


#ifdef EXTERNAL_EOS
using namespace EOS_Toolkit;
#endif

class Commons {
public:

	/* 
	 * ReprimAnd EOS
	 * External library based on https://arxiv.org/abs/2005.01821
	 * Requires installation of the library at https://zenodo.org/record/4075317#.X9nVJ8J7nJk
	 */
	//Variables for external equation of state calculation using RePrimAnd
	#ifdef EXTERNAL_EOS
    class ExternalEos {
    public:
    	static eos_thermal eos;
    	static con2prim_mhd*  cv2pv;
    	static int reprimand_eos_type;
		static double reprimand_c2p_acc;
		static double reprimand_atmo_Ye;
		static double reprimand_max_z;
		static double reprimand_max_b;
		static double reprimand_atmo_rho;
		static double reprimand_rho_strict;
		static double reprimand_gamma_th;
		static double reprimand_max_rho;
		static double reprimand_max_eps;
    };
	#endif


    static void initialization() {
    	#ifdef EXTERNAL_EOS
    	// Ideal Gas
    	if (ExternalEos::reprimand_eos_type == 0) {
    		double adiabatic_index = 1 / (ExternalEos::reprimand_gamma_th - 1);
	  		ExternalEos::eos = make_eos_idealgas(adiabatic_index, ExternalEos::reprimand_max_eps, ExternalEos::reprimand_max_rho);
		}
		//Hybrid
		if (ExternalEos::reprimand_eos_type == 1) {
			eos_barotr eos_c = load_eos_barotr("external.eos.h5");
	  		ExternalEos::eos = make_eos_hybrid(eos_c, ExternalEos::reprimand_gamma_th, ExternalEos::reprimand_max_eps, ExternalEos::reprimand_max_rho);
	  	}
		// Tabular 
		if (ExternalEos::reprimand_eos_type == 2) {
			ExternalEos::eos = load_eos_thermal("external.eos.h5");
		}
		
		//Some values to print
		std::cout<<"range rho "<<ExternalEos::eos.range_rho().min()<<" "<<ExternalEos::eos.range_rho().max()<<std::endl;
		std::cout<<"range eps (rho min) "<<ExternalEos::eos.range_eps(0, 0).min()<<" "<<ExternalEos::eos.range_eps(0, 0).max()<<std::endl;
		std::cout<<"range eps (rho max) "<<ExternalEos::eos.range_eps(ExternalEos::eos.range_rho().max(), 0).min()<<" "<<ExternalEos::eos.range_eps(ExternalEos::eos.range_rho().max(), 0).max()<<std::endl;

	  	//Set up atmosphere
		real_t atmo_eps = ExternalEos::eos.range_eps(ExternalEos::reprimand_atmo_rho, ExternalEos::reprimand_atmo_Ye).min();
   
	  	real_t atmo_cut = ExternalEos::reprimand_atmo_rho * 1.01;
	  	real_t atmo_p = ExternalEos::eos.at_rho_eps_ye(ExternalEos::reprimand_atmo_rho, atmo_eps, ExternalEos::reprimand_atmo_Ye).press();

	  	atmosphere atmo{ExternalEos::reprimand_atmo_rho, atmo_eps, ExternalEos::reprimand_atmo_Ye, atmo_p, atmo_cut};

	  	//Primitive recovery parameters
	  	bool  ye_lenient = false;
	  	int max_iter = 40;
		ExternalEos::cv2pv = new con2prim_mhd(ExternalEos::eos, ExternalEos::reprimand_rho_strict, ye_lenient, ExternalEos::reprimand_max_z, ExternalEos::reprimand_max_b, atmo, ExternalEos::reprimand_c2p_acc, max_iter);
		#endif
    }

};

namespace external {

int externalCon2prim(double& parEfu_x, double& parEfu_y, double& parEfu_z, double& parrhof, double& parpf, double& parepsf, double& parvfd_x, double& parvfd_y, double& parvfd_z, double& parsqcs, double& pardpfdrho, double& pardpfdeps, double parDf, double& parSfd_x, double& parSfd_y, double& parSfd_z, double& partauf, double& parW, double parchi, double pargtu_xx, double pargtu_xy, double pargtu_xz, double pargtu_yy, double pargtu_yz, double pargtu_zz, double pargtd_xx, double pargtd_xy, double pargtd_xz, double pargtd_yy, double pargtd_yz, double pargtd_zz, double parBfu_x, double parBfu_y, double parBfu_z, const double* dx, const double simPlat_dt, const int ilast, const int jlast);
}


#endif
