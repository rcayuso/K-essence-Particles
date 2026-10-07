
#include "Commons.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef EXTERNAL_EOS
eos_thermal Commons::ExternalEos::eos;
con2prim_mhd * Commons::ExternalEos::cv2pv;
double Commons::ExternalEos::reprimand_c2p_acc;
double Commons::ExternalEos::reprimand_atmo_Ye;
double Commons::ExternalEos::reprimand_max_z;
double Commons::ExternalEos::reprimand_max_b;
double Commons::ExternalEos::reprimand_atmo_rho;
double Commons::ExternalEos::reprimand_rho_strict;
double Commons::ExternalEos::reprimand_gamma_th;
double Commons::ExternalEos::reprimand_max_rho;
double Commons::ExternalEos::reprimand_max_eps;
int Commons::ExternalEos::reprimand_eos_type;
        

/* 
 * ReprimAnd EOS
 * External library based on https://arxiv.org/abs/2005.01821
 * Requires installation of the library at https://zenodo.org/record/4075317#.X9nVJ8J7nJk
 */
int external::externalCon2prim(double& parEfu_x, double& parEfu_y, double& parEfu_z, double& parrhof, double& parpf, double& parepsf, double& parvfd_x, double& parvfd_y, double& parvfd_z, double& parsqcs, double& pardpfdrho, double& pardpfdeps, double parDf, double& parSfd_x, double& parSfd_y, double& parSfd_z, double& partauf, double& parW, double parchi, double pargtu_xx, double pargtu_xy, double pargtu_xz, double pargtu_yy, double pargtu_yz, double pargtu_zz, double pargtd_xx, double pargtd_xy, double pargtd_xz, double pargtd_yy, double pargtd_yz, double pargtd_zz, double parBfu_x, double parBfu_y, double parBfu_z, const double* dx, const double simPlat_dt, const int ilast, const int jlast) {

    //Collect variables
    cons_vars_mhd evolved{parDf, partauf, 0.5 * parDf, {parSfd_x,parSfd_y,parSfd_z}, {parBfu_x,parBfu_y,parBfu_z}};    

    //Create metric variable
    sm_tensor2_sym<double, 3, false, false> lo(pargtd_xx/parchi, pargtd_xy/parchi, pargtd_yy/parchi, pargtd_xz/parchi, pargtd_yz/parchi, pargtd_zz/parchi);
    sm_metric3 g((sm_metric<double, 3>) lo);

    prim_vars_mhd primitives;
    con2prim_mhd::report rep;

    //Conservative to primitive variables
    (*Commons::ExternalEos::cv2pv)(primitives, evolved, g, rep);

    if (rep.failed()) {
        std::cerr << rep.debug_message()<<std::endl; 
        return -1;
    }
    //Conservative variables were adjusted
    if (rep.adjust_cons) {
        parSfd_x = evolved.scon(0);
        parSfd_y = evolved.scon(1);
        parSfd_z = evolved.scon(2);
        partauf = evolved.tau;
        parDf = evolved.dens;
        parBfu_x = evolved.bcons(0);
        parBfu_y = evolved.bcons(1);
        parBfu_z = evolved.bcons(2);
    }
    //Primitive variables assignment
    parrhof = primitives.rho;
    parepsf = primitives.eps;
    parpf = primitives.press;
    parW = primitives.w_lor;

    double vux = primitives.vel(0);
    double vuy = primitives.vel(1);
    double vuz = primitives.vel(2);

    parvfd_x = (pargtd_xx * vux + pargtd_xy * vuy + pargtd_xz * vuz) / parchi;
    parvfd_y = (pargtd_xy * vux + pargtd_yy * vuy + pargtd_yz * vuz) / parchi;
    parvfd_z = (pargtd_xz * vux + pargtd_yz * vuy + pargtd_zz * vuz) / parchi;
    parEfu_x = primitives.E(0);
    parEfu_y = primitives.E(1);
    parEfu_z = primitives.E(2);

    pardpfdrho = Commons::ExternalEos::eos.at_rho_eps_ye(primitives.rho, primitives.eps, primitives.ye).dpress_drho();
    pardpfdeps = Commons::ExternalEos::eos.at_rho_eps_ye(primitives.rho, primitives.eps, primitives.ye).dpress_deps();
    parsqcs = Commons::ExternalEos::eos.at_rho_eps_ye(primitives.rho, primitives.eps, primitives.ye).csnd();

    return  1;
}

#endif

#ifndef EXTERNAL_EOS
int external::externalCon2prim(double& parEfu_x, double& parEfu_y, double& parEfu_z, double& parrhof, double& parpf, double& parepsf, double& parvfd_x, double& parvfd_y, double& parvfd_z, double& parsqcs, double& pardpfdrho, double& pardpfdeps, double parDf, double& parSfd_x, double& parSfd_y, double& parSfd_z, double& partauf, double& parW, double parchi, double pargtu_xx, double pargtu_xy, double pargtu_xz, double pargtu_yy, double pargtu_yz, double pargtu_zz, double pargtd_xx, double pargtd_xy, double pargtd_xz, double pargtd_yy, double pargtd_yz, double pargtd_zz, double parBfu_x, double parBfu_y, double parBfu_z, const double* dx, const double simPlat_dt, const int ilast, const int jlast) {

    printf("Error in externalCon2prim: compile using Reprimand to use externalCon2prim\n");
    exit(-1);

}
#endif
