/*************************************************************************
 *
 * This file is part of the SAMRAI distribution.  For full copyright
 * information, see COPYRIGHT and COPYING.LESSER.
 *
 * Copyright:     (c) 1997-2016 Lawrence Livermore National Security, LLC
 * Description:   Conservative linear refine operator for cell-centered
 *                double data on a Skeleton mesh.
 *
 ************************************************************************/

#include "LagrangianPolynomicRefine.h"

#include <float.h>
#include <math.h>
#include "SAMRAI/hier/Index.h"
#include "SAMRAI/pdat/NodeData.h"
#include "SAMRAI/pdat/NodeVariable.h"
#include "SAMRAI/tbox/Utilities.h"
#include "SAMRAI/tbox/TimerManager.h"

#define vector3D(v, i, j, k) (v)[i+ilast*(j+jlast*(k))]
#define vector2D(v, i, j) (v)[i+ilast*(j)]
#define vector3DC(v, i, j, k) (v)[i+ilastC*(j+jlastC*(k))]
#define vector2DC(v, i, j) (v)[i+ilastC*(j)]

#define POLINT_MACRO_LINEAR(y1, y2, R, p) ( (R*y1-p*y1+p*y2)/R )
#define POLINT_MACRO_CUBIC(y1, y2, y3, y4, R, p) ( (1.0/6)*(6*R*R*R*y2-2*R*R*p*y1-3*R*R*p*y2+6*R*R*p*y3-R*R*p*y4+3*R*p*p*y1-6*R*p*p*y2+3*R*p*p*y3-p*p*p*y1+3*p*p*p*y2-3*p*p*p*y3+p*p*p*y4)/(R*R*R) )
#define POLINT_MACRO_QUINTIC(y1, y2, y3, y4, y5, y6, R, p) ( (1.0/120)*(120*R*R*R*R*R*y3+6*R*R*R*R*p*y1-60*R*R*R*R*p*y2-40*R*R*R*R*p*y3+120*R*R*R*R*p*y4-30*R*R*R*R*p*y5+4*R*R*R*R*p*y6-5*R*R*R*p*p*y1+80*R*R*R*p*p*y2-150*R*R*R*p*p*y3+80*R*R*R*p*p*y4-5*R*R*R*p*p*y5-5*R*R*p*p*p*y1-5*R*R*p*p*p*y2+50*R*R*p*p*p*y3-70*R*R*p*p*p*y4+35*R*R*p*p*p*y5-5*R*R*p*p*p*y6+5*R*p*p*p*p*y1-20*R*p*p*p*p*y2+30*R*p*p*p*p*y3-20*R*p*p*p*p*y4+5*R*p*p*p*p*y5-p*p*p*p*p*y1+5*p*p*p*p*p*y2-10*p*p*p*p*p*y3+10*p*p*p*p*p*y4-5*p*p*p*p*p*y5+p*p*p*p*p*y6)/(R*R*R*R*R) )

#define POLINT_MACRO_LINEAR_R2(y1, y2) ( 0.5 * y1 + 0.5 * y2 )
#define POLINT_MACRO_CUBIC_R2(y1, y2, y3, y4) (0.5625*(y2 + y3)-0.0625*(y1 + y4))
#define POLINT_MACRO_QUINTIC_R2(y1, y2, y3, y4, y5, y6) (0.5859375 * (y3 + y4) - 0.09765625 * (y2 + y5) + 0.01171875 * (y1+y6))

#define POLINT_MACRO_CUBIC_RIGHT(y1, y2, y3, y4, R, p) (-(1.0/6)*(p*p*p*y1-3*p*p*p*y2+3*p*p*p*y3-p*p*p*y4-6*p*p*R*y1+15*p*p*R*y2-12*p*p*R*y3+3*p*p*R*y4+11*p*R*R*y1-18*p*R*R*y2+9*p*R*R*y3-2*p*R*R*y4-6*R*R*R*y1)/(R*R*R))
#define POLINT_MACRO_CUBIC_LEFT(y1, y2, y3, y4, R, p) ((1.0/6)*(6*R*R*R*y3+R*R*p*y1-6*R*R*p*y2+3*R*R*p*y3+2*R*R*p*y4+3*R*p*p*y2-6*R*p*p*y3+3*R*p*p*y4-p*p*p*y1+3*p*p*p*y2-3*p*p*p*y3+p*p*p*y4)/(R*R*R))

#define POLINT_MACRO_CUBIC_RIGHT_R2(y1, y2, y3, y4) (0.3125 * y1 + 0.9375 * y2 - 0.3125 * y3 + 0.0625 * y4)
#define POLINT_MACRO_CUBIC_LEFT_R2(y1, y2, y3, y4) (0.3125 * y4 + 0.9375 * y3 - 0.3125 * y2 + 0.0625 * y1)

#define DATA2D(i, j) (yat[ibase + i + nrows*(jbase + j)])
#define DATA3D(i, j, k) (yat[ibase + i + nrows*(jbase + j + ncolumns*(kbase + k))])

using namespace SAMRAI;

std::shared_ptr<tbox::Timer> LagrangianPolynomicRefine::t_interpolate;

LagrangianPolynomicRefine::
LagrangianPolynomicRefine(
   const bool refine_boundary,                                    //This parameters enable/disable the interpolation in the nodes just on the boundary between internal patch and ghost zone. When creating a new level from regridding, the boundary must be interpolated, otherwise is not necessary
   const int order,
   std::shared_ptr<hier::PatchHierarchy >& patch_hierarchy,
   const tbox::Dimension& dim):
   hier::RefineOperator("LAGRANGIAN_POLYNOMIC_REFINE"),
   d_stencil(order + 1),
   d_refine_boundary(refine_boundary),
   d_patch_hierarchy(patch_hierarchy),
   d_dim(dim)
{
   TBOX_ASSERT(order == 1 || order == 3 || order == 5);
   d_half_stencil = (order + 1)/2;

   t_interpolate = tbox::TimerManager::getManager()-> getTimer("Spatial refinement");
}

LagrangianPolynomicRefine::~
LagrangianPolynomicRefine()
{
}

int
LagrangianPolynomicRefine::getOperatorPriority() const
{
   return 0;
}

hier::IntVector
LagrangianPolynomicRefine::getStencilWidth(const tbox::Dimension& dim) const {
   return hier::IntVector(dim, d_half_stencil);
}

void LagrangianPolynomicRefine::refine(
   hier::Patch& fine,
   const hier::Patch& coarse,
   const int dst_component,
   const int src_component,
   const hier::BoxOverlap& fine_overlap,
   const hier::IntVector& ratio) const
{
   t_interpolate->start();
   const pdat::NodeOverlap* t_overlap =
      CPP_CAST<const pdat::NodeOverlap *>(&fine_overlap);

   TBOX_ASSERT(t_overlap != 0);
   TBOX_ASSERT(fine.getDim() == tbox::Dimension(2) || fine.getDim() == tbox::Dimension(3));

   const hier::BoxContainer& boxes = t_overlap->getDestinationBoxContainer();

   if (d_stencil == 2) {
      for (hier::BoxContainer::const_iterator b = boxes.begin();
           b != boxes.end(); ++b) {
         refine_linear(fine,
            coarse,
            dst_component,
            src_component,
            *b,
            ratio);
      }
   }
   if (d_stencil == 4) {
      for (hier::BoxContainer::const_iterator b = boxes.begin();
           b != boxes.end(); ++b) {
         refine_cubic(fine,
            coarse,
            dst_component,
            src_component,
            *b,
            ratio);
      }
   }
   if (d_stencil == 6) {
      for (hier::BoxContainer::const_iterator b = boxes.begin();
           b != boxes.end(); ++b) {
         refine_quintic(fine,
            coarse,
            dst_component,
            src_component,
            *b,
            ratio);
      }
   }
   t_interpolate->stop();
}

void LagrangianPolynomicRefine::refine_linear(
   hier::Patch& fine,
   const hier::Patch& coarse,
   const int dst_component,
   const int src_component,
   const hier::Box& fine_box,
   const hier::IntVector& ratio) const
{
   std::shared_ptr<pdat::NodeData<double> > cdata(
      SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(
         coarse.getPatchData(src_component)));
   std::shared_ptr<pdat::NodeData<double> > fdata(
      SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(
         fine.getPatchData(dst_component)));

   TBOX_ASSERT(cdata);
   TBOX_ASSERT(fdata);
   TBOX_ASSERT(cdata->getDepth() == fdata->getDepth());

   const hier::Index cilo = cdata->getGhostBox().lower();
   const hier::Index cihi = cdata->getGhostBox().upper();
   const hier::Index filo = fdata->getGhostBox().lower();
   const hier::Index fihi = fdata->getGhostBox().upper();

   const hier::Index il = fdata->getBox().lower();
   const hier::Index ih = fdata->getBox().upper();

   const hier::Index ifirstf = fine_box.lower();
   const hier::Index ilastf = fine_box.upper();

   double ymtmp[2];
   double yntmp[2];

   std::vector<double> ya;
   int kbase, jbase, ibase; 

   int Rx = ratio[0];
   int Ry = ratio[1];

   for (int d = 0; d < fdata->getDepth(); ++d) {

      double* fieldf = fdata->getPointer(d);
      double* fieldc = cdata->getPointer(d);

      if (fine.getDim() == tbox::Dimension(2)) {
         int ilast = fihi(0)-filo(0) + 2;

         int ilastC = cihi(0)-cilo(0) + 2;

         double ratio0f = (double) ratio[0];
         double ratio1f = (double) ratio[1];

         int nrows = (floor(ilastf[0] / ratio0f) + d_half_stencil) - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1) + 1;
         //Initialization of coarser value array
         for(int j = floor(ifirstf[1] / ratio1f) - d_half_stencil + 1; j <= floor(ilastf[1] / ratio1f) + d_half_stencil; j++) {
            for(int i = floor(ifirstf[0] / ratio0f) - d_half_stencil + 1; i <= floor(ilastf[0] / ratio0f) + d_half_stencil; i++) {
               ya.push_back(vector2DC(fieldc, i - cilo[0], j - cilo[1]));
            }
         }
         double *yat = &ya[0];
         //Iterate over fine nodes and interpolate
         for(int j = ifirstf[1]; j <= ilastf[1]; j++) {
            jbase = floor(j / ratio1f) - d_half_stencil + 1 - (floor(ifirstf[1] / ratio1f) - d_half_stencil + 1);
            int py = abs(j % Ry);
            bool y_alingment = py == 0;
            for(int i = ifirstf[0]; i <= ilastf[0]; i++) {
               ibase = floor(i / ratio0f) - d_half_stencil + 1 - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1);
               int px = abs(i % Rx);
               bool x_alingment = px == 0;
               if (d_refine_boundary || i < il[0] || i > ih[0]+1 || j < il[1] || j > ih[1] + 1) {
                  if (x_alingment) {
                     if (y_alingment) {
                        vector2D(fieldf, i - filo[0], j - filo[1]) = DATA2D(0, 0);
                     }
                     else {
                        if (Ry > 2) {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_LINEAR(DATA2D(0, 0), DATA2D(0, 1), Ry, py);
                        }
                        else {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_LINEAR_R2(DATA2D(0, 0), DATA2D(0, 1));   
                        }
                     }
                  }
                  else {
                     if (y_alingment) {
                        if (Rx > 2) {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_LINEAR(DATA2D(0, 0), DATA2D(1, 0), Rx, px); 
                        }
                        else {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_LINEAR_R2(DATA2D(0, 0), DATA2D(1, 0)); 
                        }
                     }
                     else {
                        if (Rx > 2) {
                           ymtmp[0] = POLINT_MACRO_LINEAR(DATA2D(0, 0), DATA2D(1, 0), Rx, px);
                           ymtmp[1] = POLINT_MACRO_LINEAR(DATA2D(0, 1), DATA2D(1, 1), Rx, px);
                        }
                        else {
                           ymtmp[0] = POLINT_MACRO_LINEAR_R2(DATA2D(0, 0), DATA2D(1, 0));
                           ymtmp[1] = POLINT_MACRO_LINEAR_R2(DATA2D(0, 1), DATA2D(1, 1));
                        }                        
                        if (Ry > 2) {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_LINEAR(ymtmp[0], ymtmp[1], Ry, py);
                        }
                        else {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_LINEAR_R2(ymtmp[0], ymtmp[1]);
                        }
                     }
                  }
               }
            }
         }
      } else if (fine.getDim() == tbox::Dimension(3)) {
         int Rz = ratio[2];
         int ilast = fihi(0)-filo(0) + 2;
         int jlast = fihi(1)-filo(1) + 2;

         int ilastC = cihi(0)-cilo(0) + 2;
         int jlastC = cihi(1)-cilo(1) + 2;

         double ratio0f = (double) ratio[0];
         double ratio1f = (double) ratio[1];
         double ratio2f = (double) ratio[2];

         int nrows = (floor(ilastf[0] / ratio0f) + d_half_stencil) - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1) + 1;
         int ncolumns = (floor(ilastf[1] / ratio1f) + d_half_stencil) - (floor(ifirstf[1] / ratio1f) - d_half_stencil + 1) + 1;
         //Initialization of coarser value array
         for(int k = floor(ifirstf[2] / ratio2f) - d_half_stencil + 1; k <= floor(ilastf[2] / ratio2f) + d_half_stencil; k++) {
            for(int j = floor(ifirstf[1] / ratio1f) - d_half_stencil + 1; j <= floor(ilastf[1] / ratio1f) + d_half_stencil; j++) {
               for(int i = floor(ifirstf[0] / ratio0f) - d_half_stencil + 1; i <= floor(ilastf[0] / ratio0f) + d_half_stencil; i++) {
                  ya.push_back(vector3DC(fieldc, i - cilo[0], j - cilo[1], k - cilo[2]));
               }
            }
         }
         double *yat = &ya[0];
         //Iterate over fine nodes and interpolate
         for(int k = ifirstf[2]; k <= ilastf[2]; k++) {
            kbase = floor(k / ratio2f) - d_half_stencil + 1 - (floor(ifirstf[2] / ratio2f) - d_half_stencil + 1);
            int pz = abs(k % Rz);
            bool z_alingment = pz == 0;
            //int kparent = k/2 - cilo[2];
            for(int j = ifirstf[1]; j <= ilastf[1]; j++) {
               jbase = floor(j / ratio1f) - d_half_stencil + 1 - (floor(ifirstf[1] / ratio1f) - d_half_stencil + 1);
               int py = abs(j % Ry);
               bool y_alingment = py == 0;
               //int jparent = j/2 - cilo[1];
               for(int i = ifirstf[0]; i <= ilastf[0]; i++) {
                  ibase = floor(i / ratio0f) - d_half_stencil + 1 - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1);
                  int px = abs(i % Rx);
                  bool x_alingment = px == 0;
                  //int iparent = i/2 - cilo[0];
                  if (d_refine_boundary || i < il[0] || i > ih[0]+1 || j < il[1] || j > ih[1] + 1 || k < il[2] || k > ih[2] + 1) {
                     if (x_alingment) {
                        if (y_alingment) {
                           if (z_alingment) {
                              vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = DATA3D(0, 0, 0);
                           }
                           else {
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR(DATA3D(0, 0, 0), DATA3D(0, 0, 1), Rz, pz);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 0), DATA3D(0, 0, 1));
                              }
                           }
                        }
                        else {
                           if (z_alingment) {
                              if (Ry > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR(DATA3D(0, 0, 0), DATA3D(0, 1, 0), Ry, py);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 0), DATA3D(0, 1, 0));
                              }
                           }
                           else {
                              if (Ry > 2) {
                                 yntmp[0] = POLINT_MACRO_LINEAR(DATA3D(0, 0, 0), DATA3D(0, 1, 0), Ry, py);
                                 yntmp[1] = POLINT_MACRO_LINEAR(DATA3D(0, 0, 1), DATA3D(0, 1, 1), Ry, py);
                              }
                              else {
                                 yntmp[0] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 0), DATA3D(0, 1, 0));
                                 yntmp[1] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 1), DATA3D(0, 1, 1));
                              }                              
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR(yntmp[0],yntmp[1], Rz, pz);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR_R2(yntmp[0],yntmp[1]);
                              }
                           }
                        }
                     }
                     else {
                        if (y_alingment) {
                           if (z_alingment) {
                              if (Rx > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR(DATA3D(0, 0, 0), DATA3D(1, 0, 0), Rx, px);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 0), DATA3D(1, 0, 0));
                              }
                           }
                           else {
                              if (Rx > 2) {
                                 yntmp[0] = POLINT_MACRO_LINEAR(DATA3D(0, 0, 0), DATA3D(1, 0, 0), Rx, px);
                                 yntmp[1] = POLINT_MACRO_LINEAR(DATA3D(0, 0, 1), DATA3D(1, 0, 1), Rx, px);
                              }
                              else {
                                 yntmp[0] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 0), DATA3D(1, 0, 0));
                                 yntmp[1] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 1), DATA3D(1, 0, 1));
                              }
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR(yntmp[0],yntmp[1], Rz, pz);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR_R2(yntmp[0],yntmp[1]);
                              }
                           }
                    
                        }
                        else {
                           if (z_alingment) {
                              if (Rx > 2) {
                                 yntmp[0] = POLINT_MACRO_LINEAR(DATA3D(0, 0, 0), DATA3D(1, 0, 0), Rx, px);
                                 yntmp[1] = POLINT_MACRO_LINEAR(DATA3D(0, 1, 0), DATA3D(1, 1, 0), Rx, px);
                              }
                              else {
                                 yntmp[0] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 0), DATA3D(1, 0, 0));
                                 yntmp[1] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 1, 0), DATA3D(1, 1, 0));
                              }
                              if (Ry > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR(yntmp[0],yntmp[1], Ry, py);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR_R2(yntmp[0],yntmp[1]);
                              }
                           }
                           else {
                              if (Ry > 2) {
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_LINEAR(DATA3D(0, 0, 0), DATA3D(1, 0, 0), Rx, px);
                                    yntmp[1] = POLINT_MACRO_LINEAR(DATA3D(0, 1, 0), DATA3D(1, 1, 0), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 0), DATA3D(1, 0, 0));
                                    yntmp[1] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 1, 0), DATA3D(1, 1, 0));
                                 }                           
                                 ymtmp[0] = POLINT_MACRO_LINEAR(yntmp[0],yntmp[1], Ry, py);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_LINEAR(DATA3D(0, 0, 1), DATA3D(1, 0, 1), Rx, px);
                                    yntmp[1] = POLINT_MACRO_LINEAR(DATA3D(0, 1, 1), DATA3D(1, 1, 1), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 1), DATA3D(1, 0, 1));
                                    yntmp[1] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 1, 1), DATA3D(1, 1, 1));
                                 }                           
                                 ymtmp[1] = POLINT_MACRO_LINEAR(yntmp[0],yntmp[1], Ry, py);
                              }
                              else{
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_LINEAR(DATA3D(0, 0, 0), DATA3D(1, 0, 0), Rx, px);
                                    yntmp[1] = POLINT_MACRO_LINEAR(DATA3D(0, 1, 0), DATA3D(1, 1, 0), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 0), DATA3D(1, 0, 0));
                                    yntmp[1] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 1, 0), DATA3D(1, 1, 0));
                                 }
                                 ymtmp[0] = POLINT_MACRO_LINEAR_R2(yntmp[0],yntmp[1]);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_LINEAR(DATA3D(0, 0, 1), DATA3D(1, 0, 1), Rx, px);
                                    yntmp[1] = POLINT_MACRO_LINEAR(DATA3D(0, 1, 1), DATA3D(1, 1, 1), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 0, 1), DATA3D(1, 0, 1));
                                    yntmp[1] = POLINT_MACRO_LINEAR_R2(DATA3D(0, 1, 1), DATA3D(1, 1, 1));
                                 }
                                 ymtmp[1] = POLINT_MACRO_LINEAR_R2(yntmp[0],yntmp[1]);
                              }
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR(ymtmp[0],ymtmp[1], Rz, pz);   
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_LINEAR_R2(ymtmp[0],ymtmp[1]);
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      } else {
         TBOX_ERROR("LagrangianPolynomicRefine error...\n"
            << "dimension > 3 or < 2 not supported." << endl);

      }
   }
}

void LagrangianPolynomicRefine::refine_cubic(
   hier::Patch& fine,
   const hier::Patch& coarse,
   const int dst_component,
   const int src_component,
   const hier::Box& fine_box,
   const hier::IntVector& ratio) const
{
   std::shared_ptr<pdat::NodeData<double> > cdata(
      SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(
         coarse.getPatchData(src_component)));
   std::shared_ptr<pdat::NodeData<double> > fdata(
      SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(
         fine.getPatchData(dst_component)));

   TBOX_ASSERT(cdata);
   TBOX_ASSERT(fdata);
   TBOX_ASSERT(cdata->getDepth() == fdata->getDepth());

   const hier::Index cilo = cdata->getGhostBox().lower();
   const hier::Index cihi = cdata->getGhostBox().upper();
   const hier::Index filo = fdata->getGhostBox().lower();
   const hier::Index fihi = fdata->getGhostBox().upper();

   const hier::Index il = fdata->getBox().lower();
   const hier::Index ih = fdata->getBox().upper();

   const hier::Index ifirstf = fine_box.lower();
   const hier::Index ilastf = fine_box.upper();

   int Rx = ratio[0];
   int Ry = ratio[1];

   std::vector<double> ya;
   int kbase, jbase, ibase; 

   double ymtmp[4];
   double yntmp[4];

   for (int d = 0; d < fdata->getDepth(); ++d) {

      double* fieldf = fdata->getPointer(d);
      double* fieldc = cdata->getPointer(d);

      if (fine.getDim() == tbox::Dimension(2)) {
         int ilast = fihi(0)-filo(0) + 2;

         int ilastC = cihi(0)-cilo(0) + 2;

         double ratio0f = (double) ratio[0];
         double ratio1f = (double) ratio[1];

         int nrows = (floor(ilastf[0] / ratio0f) + d_half_stencil) - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1) + 1;
         //Initialization of coarser value array
         for(int j = floor(ifirstf[1] / ratio1f) - d_half_stencil + 1; j <= floor(ilastf[1] / ratio1f) + d_half_stencil; j++) {
            for(int i = floor(ifirstf[0] / ratio0f) - d_half_stencil + 1; i <= floor(ilastf[0] / ratio0f) + d_half_stencil; i++) {
               ya.push_back(vector2DC(fieldc, i - cilo[0], j - cilo[1]));
            }
         }
         double *yat = &ya[0];
         //Iterate over fine nodes and interpolate
         for(int j = ifirstf[1]; j <= ilastf[1]; j++) {
            jbase = floor(j / ratio1f) - d_half_stencil + 1 - (floor(ifirstf[1] / ratio1f) - d_half_stencil + 1);
            int py = abs(j % Ry);
            bool y_alingment = py == 0;
            for(int i = ifirstf[0]; i <= ilastf[0]; i++) {
               ibase = floor(i / ratio0f) - d_half_stencil + 1 - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1);
               int px = abs(i % Rx);
               bool x_alingment = px == 0;
               if (d_refine_boundary || i < il[0] || i > ih[0]+1 || j < il[1] || j > ih[1] + 1) {
                  if (x_alingment) {
                     if (y_alingment) {
                        vector2D(fieldf, i - filo[0], j - filo[1]) = DATA2D(1, 1);
                     }
                     else {
                        if (Ry > 2) {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_CUBIC(DATA2D(1, 0), DATA2D(1, 1), DATA2D(1, 2), DATA2D(1, 3), Ry, py);
                        }
                        else {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_CUBIC_R2(DATA2D(1, 0), DATA2D(1, 1), DATA2D(1, 2), DATA2D(1, 3));
                        }
                     }
                  }
                  else {
                     if (y_alingment) {
                        if (Rx > 2) {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_CUBIC(DATA2D(0, 1), DATA2D(1, 1), DATA2D(2, 1), DATA2D(3, 1), Rx, px);
                        }
                        else {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_CUBIC_R2(DATA2D(0, 1), DATA2D(1, 1), DATA2D(2, 1), DATA2D(3, 1));
                        }
                     }
                     else {
                        if (Rx > 2) {
                           ymtmp[0] = POLINT_MACRO_CUBIC(DATA2D(0, 0), DATA2D(1, 0),DATA2D(2, 0), DATA2D(3, 0), Rx, px);
                           ymtmp[1] = POLINT_MACRO_CUBIC(DATA2D(0, 1), DATA2D(1, 1),DATA2D(2, 1), DATA2D(3, 1), Rx, px);
                           ymtmp[2] = POLINT_MACRO_CUBIC(DATA2D(0, 2), DATA2D(1, 2),DATA2D(2, 2), DATA2D(3, 2), Rx, px);
                           ymtmp[3] = POLINT_MACRO_CUBIC(DATA2D(0, 3), DATA2D(1, 3),DATA2D(2, 3), DATA2D(3, 3), Rx, px);
                        }
                        else {
                           ymtmp[0] = POLINT_MACRO_CUBIC_R2(DATA2D(0, 0), DATA2D(1, 0),DATA2D(2, 0), DATA2D(3, 0));
                           ymtmp[1] = POLINT_MACRO_CUBIC_R2(DATA2D(0, 1), DATA2D(1, 1),DATA2D(2, 1), DATA2D(3, 1));
                           ymtmp[2] = POLINT_MACRO_CUBIC_R2(DATA2D(0, 2), DATA2D(1, 2),DATA2D(2, 2), DATA2D(3, 2));
                           ymtmp[3] = POLINT_MACRO_CUBIC_R2(DATA2D(0, 3), DATA2D(1, 3),DATA2D(2, 3), DATA2D(3, 3));
                        }
                        if (Ry > 2) {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_CUBIC(ymtmp[0], ymtmp[1], ymtmp[2], ymtmp[3], Ry, py);
                        }
                        else {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_CUBIC_R2(ymtmp[0], ymtmp[1], ymtmp[2], ymtmp[3]);
                        }
                     }
                  }
               }
            }
         }
      } else if (fine.getDim() == tbox::Dimension(3)) {
         int Rz = ratio[2];

         int ilast = fihi(0)-filo(0) + 2;
         int jlast = fihi(1)-filo(1) + 2;

         int ilastC = cihi(0)-cilo(0) + 2;
         int jlastC = cihi(1)-cilo(1) + 2;

         double ratio0f = (double) ratio[0];
         double ratio1f = (double) ratio[1];
         double ratio2f = (double) ratio[2];

         int nrows = (floor(ilastf[0] / ratio0f) + d_half_stencil) - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1) + 1;
         int ncolumns = (floor(ilastf[1] / ratio1f) + d_half_stencil) - (floor(ifirstf[1] / ratio1f) - d_half_stencil + 1) + 1;
         //Initialization of coarser value array
         for(int k = floor(ifirstf[2] / ratio2f) - d_half_stencil + 1; k <= floor(ilastf[2] / ratio2f) + d_half_stencil; k++) {
            for(int j = floor(ifirstf[1] / ratio1f) - d_half_stencil + 1; j <= floor(ilastf[1] / ratio1f) + d_half_stencil; j++) {
               for(int i = floor(ifirstf[0] / ratio0f) - d_half_stencil + 1; i <= floor(ilastf[0] / ratio0f) + d_half_stencil; i++) {
                  ya.push_back(vector3DC(fieldc, i - cilo[0], j - cilo[1], k - cilo[2]));
               }
            }
         }
         double *yat = &ya[0];
         //Iterate over fine nodes and interpolate
         for(int k = ifirstf[2]; k <= ilastf[2]; k++) {
            kbase = floor(k / ratio2f) - d_half_stencil + 1 - (floor(ifirstf[2] / ratio2f) - d_half_stencil + 1);
            int pz = abs(k % Rz);
            bool z_alingment = pz == 0;
            for(int j = ifirstf[1]; j <= ilastf[1]; j++) {
               jbase = floor(j / ratio1f) - d_half_stencil + 1 - (floor(ifirstf[1] / ratio1f) - d_half_stencil + 1);
               int py = abs(j % Ry);
               bool y_alingment = py == 0;
               for(int i = ifirstf[0]; i <= ilastf[0]; i++) {
                  ibase = floor(i / ratio0f) - d_half_stencil + 1 - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1);
                  int px = abs(i % Rx);
                  bool x_alingment = px == 0;
                  if (d_refine_boundary || i < il[0] || i > ih[0]+1 || j < il[1] || j > ih[1] + 1 || k < il[2] || k > ih[2] + 1) {
                     if (x_alingment) {
                        if (y_alingment) {
                           if (z_alingment) {
                              vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = DATA3D(1, 1, 1);
                           }
                           else {
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC(DATA3D(1, 1, 0), DATA3D(1, 1, 1), DATA3D(1, 1, 2), DATA3D(1, 1, 3), Rz, pz);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC_R2(DATA3D(1, 1, 0), DATA3D(1, 1, 1), DATA3D(1, 1, 2), DATA3D(1, 1, 3));
                              }
                           }
                        }
                        else {
                           if (z_alingment) {
                              if (Ry > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC(DATA3D(1, 0, 1), DATA3D(1, 1, 1), DATA3D(1, 2, 1), DATA3D(1, 3, 1), Ry, py);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC_R2(DATA3D(1, 0, 1), DATA3D(1, 1, 1), DATA3D(1, 2, 1), DATA3D(1, 3, 1));   
                              }
                           }
                           else {
                              if (Ry > 2) {
                                 yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(1, 0, 0), DATA3D(1, 1, 0), DATA3D(1, 2, 0), DATA3D(1, 3, 0), Ry, py);
                                 yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(1, 0, 1), DATA3D(1, 1, 1), DATA3D(1, 2, 1), DATA3D(1, 3, 1), Ry, py);
                                 yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(1, 0, 2), DATA3D(1, 1, 2), DATA3D(1, 2, 2), DATA3D(1, 3, 2), Ry, py);
                                 yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(1, 0, 3), DATA3D(1, 1, 3), DATA3D(1, 2, 3), DATA3D(1, 3, 3), Ry, py);
                              }
                              else {
                                 yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(1, 0, 0), DATA3D(1, 1, 0), DATA3D(1, 2, 0), DATA3D(1, 3, 0));
                                 yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(1, 0, 1), DATA3D(1, 1, 1), DATA3D(1, 2, 1), DATA3D(1, 3, 1));
                                 yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(1, 0, 2), DATA3D(1, 1, 2), DATA3D(1, 2, 2), DATA3D(1, 3, 2));
                                 yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(1, 0, 3), DATA3D(1, 1, 3), DATA3D(1, 2, 3), DATA3D(1, 3, 3));
                              }
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC(yntmp[0],yntmp[1],yntmp[2],yntmp[3], Rz, pz);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC_R2(yntmp[0],yntmp[1],yntmp[2],yntmp[3]);  
                              }
                           }
                        }
                     }
                     else {
                        if (y_alingment) {
                           if (z_alingment) {
                              if (Rx > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1), Rx, px);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1));
                              }
                           }
                           else {
                              if (Rx > 2) {
                                 yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 0), DATA3D(1, 1, 0), DATA3D(2, 1, 0), DATA3D(3, 1, 0), Rx, px);
                                 yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1), Rx, px);
                                 yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2), Rx, px);
                                 yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 3), DATA3D(1, 1, 3), DATA3D(2, 1, 3), DATA3D(3, 1, 3), Rx, px);
                              }
                              else{
                                 yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 0), DATA3D(1, 1, 0), DATA3D(2, 1, 0), DATA3D(3, 1, 0));
                                 yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1));
                                 yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2));
                                 yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 3), DATA3D(1, 1, 3), DATA3D(2, 1, 3), DATA3D(3, 1, 3));
                              }
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC(yntmp[0],yntmp[1],yntmp[2],yntmp[3], Rz, pz);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC_R2(yntmp[0],yntmp[1],yntmp[2],yntmp[3]);
                              }
                           }
                        }
                        else {
                           if (z_alingment) {
                              if (Rx > 2) {
                                 yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(0, 0, 1), DATA3D(1, 0, 1), DATA3D(2, 0, 1), DATA3D(3, 0, 1), Rx, px);
                                 yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1), Rx, px);
                                 yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1), Rx, px);
                                 yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(0, 3, 1), DATA3D(1, 3, 1), DATA3D(2, 3, 1), DATA3D(3, 3, 1), Rx, px);
                              }
                              else {
                                 yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 0, 1), DATA3D(1, 0, 1), DATA3D(2, 0, 1), DATA3D(3, 0, 1));
                                 yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1));
                                 yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1));
                                 yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 3, 1), DATA3D(1, 3, 1), DATA3D(2, 3, 1), DATA3D(3, 3, 1));
                              }
                              if (Ry > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC(yntmp[0],yntmp[1],yntmp[2],yntmp[3], Ry, py);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC_R2(yntmp[0],yntmp[1],yntmp[2],yntmp[3]);
                              }
                           }
                           else {
                              if (Ry > 2) {
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(0, 0, 0), DATA3D(1, 0, 0), DATA3D(2, 0, 0), DATA3D(3, 0, 0), Rx, px);
                                    yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 0), DATA3D(1, 1, 0), DATA3D(2, 1, 0), DATA3D(3, 1, 0), Rx, px);
                                    yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(0, 2, 0), DATA3D(1, 2, 0), DATA3D(2, 2, 0), DATA3D(3, 2, 0), Rx, px);
                                    yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(0, 3, 0), DATA3D(1, 3, 0), DATA3D(2, 3, 0), DATA3D(3, 3, 0), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 0, 0), DATA3D(1, 0, 0), DATA3D(2, 0, 0), DATA3D(3, 0, 0));
                                    yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 0), DATA3D(1, 1, 0), DATA3D(2, 1, 0), DATA3D(3, 1, 0));
                                    yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 2, 0), DATA3D(1, 2, 0), DATA3D(2, 2, 0), DATA3D(3, 2, 0));
                                    yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 3, 0), DATA3D(1, 3, 0), DATA3D(2, 3, 0), DATA3D(3, 3, 0));
                                 }
                                 ymtmp[0] = POLINT_MACRO_CUBIC(yntmp[0],yntmp[1],yntmp[2],yntmp[3], Ry, py);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(0, 0, 1), DATA3D(1, 0, 1), DATA3D(2, 0, 1), DATA3D(3, 0, 1), Rx, px);
                                    yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1), Rx, px);
                                    yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1), Rx, px);
                                    yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(0, 3, 1), DATA3D(1, 3, 1), DATA3D(2, 3, 1), DATA3D(3, 3, 1), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 0, 1), DATA3D(1, 0, 1), DATA3D(2, 0, 1), DATA3D(3, 0, 1));
                                    yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1));
                                    yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1));
                                    yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 3, 1), DATA3D(1, 3, 1), DATA3D(2, 3, 1), DATA3D(3, 3, 1));
                                 }
                                 ymtmp[1] = POLINT_MACRO_CUBIC(yntmp[0],yntmp[1],yntmp[2],yntmp[3], Ry, py);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(0, 0, 2), DATA3D(1, 0, 2), DATA3D(2, 0, 2), DATA3D(3, 0, 2), Rx, px);
                                    yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2), Rx, px);
                                    yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), Rx, px);
                                    yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(0, 3, 2), DATA3D(1, 3, 2), DATA3D(2, 3, 2), DATA3D(3, 3, 2), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 0, 2), DATA3D(1, 0, 2), DATA3D(2, 0, 2), DATA3D(3, 0, 2));
                                    yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2));
                                    yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2));
                                    yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 3, 2), DATA3D(1, 3, 2), DATA3D(2, 3, 2), DATA3D(3, 3, 2));
                                 }
                                 ymtmp[2] = POLINT_MACRO_CUBIC(yntmp[0],yntmp[1],yntmp[2],yntmp[3], Ry, py);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(0, 0, 3), DATA3D(1, 0, 3), DATA3D(2, 0, 3), DATA3D(3, 0, 3), Rx, px);
                                    yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 3), DATA3D(1, 1, 3), DATA3D(2, 1, 3), DATA3D(3, 1, 3), Rx, px);
                                    yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(0, 2, 3), DATA3D(1, 2, 3), DATA3D(2, 2, 3), DATA3D(3, 2, 3), Rx, px);
                                    yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(0, 3, 3), DATA3D(1, 3, 3), DATA3D(2, 3, 3), DATA3D(3, 3, 3), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 0, 3), DATA3D(1, 0, 3), DATA3D(2, 0, 3), DATA3D(3, 0, 3));
                                    yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 3), DATA3D(1, 1, 3), DATA3D(2, 1, 3), DATA3D(3, 1, 3));
                                    yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 2, 3), DATA3D(1, 2, 3), DATA3D(2, 2, 3), DATA3D(3, 2, 3));
                                    yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 3, 3), DATA3D(1, 3, 3), DATA3D(2, 3, 3), DATA3D(3, 3, 3));
                                 }
                                 ymtmp[3] = POLINT_MACRO_CUBIC(yntmp[0],yntmp[1],yntmp[2],yntmp[3], Ry, py);
                              }
                              else {
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(0, 0, 0), DATA3D(1, 0, 0), DATA3D(2, 0, 0), DATA3D(3, 0, 0), Rx, px);
                                    yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 0), DATA3D(1, 1, 0), DATA3D(2, 1, 0), DATA3D(3, 1, 0), Rx, px);
                                    yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(0, 2, 0), DATA3D(1, 2, 0), DATA3D(2, 2, 0), DATA3D(3, 2, 0), Rx, px);
                                    yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(0, 3, 0), DATA3D(1, 3, 0), DATA3D(2, 3, 0), DATA3D(3, 3, 0), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 0, 0), DATA3D(1, 0, 0), DATA3D(2, 0, 0), DATA3D(3, 0, 0));
                                    yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 0), DATA3D(1, 1, 0), DATA3D(2, 1, 0), DATA3D(3, 1, 0));
                                    yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 2, 0), DATA3D(1, 2, 0), DATA3D(2, 2, 0), DATA3D(3, 2, 0));
                                    yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 3, 0), DATA3D(1, 3, 0), DATA3D(2, 3, 0), DATA3D(3, 3, 0));
                                 }
                                 ymtmp[0] = POLINT_MACRO_CUBIC_R2(yntmp[0],yntmp[1],yntmp[2],yntmp[3]);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(0, 0, 1), DATA3D(1, 0, 1), DATA3D(2, 0, 1), DATA3D(3, 0, 1), Rx, px);
                                    yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1), Rx, px);
                                    yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1), Rx, px);
                                    yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(0, 3, 1), DATA3D(1, 3, 1), DATA3D(2, 3, 1), DATA3D(3, 3, 1), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 0, 1), DATA3D(1, 0, 1), DATA3D(2, 0, 1), DATA3D(3, 0, 1));
                                    yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1));
                                    yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1));
                                    yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 3, 1), DATA3D(1, 3, 1), DATA3D(2, 3, 1), DATA3D(3, 3, 1));
                                 }
                                 ymtmp[1] = POLINT_MACRO_CUBIC_R2(yntmp[0],yntmp[1],yntmp[2],yntmp[3]);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(0, 0, 2), DATA3D(1, 0, 2), DATA3D(2, 0, 2), DATA3D(3, 0, 2), Rx, px);
                                    yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2), Rx, px);
                                    yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), Rx, px);
                                    yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(0, 3, 2), DATA3D(1, 3, 2), DATA3D(2, 3, 2), DATA3D(3, 3, 2), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 0, 2), DATA3D(1, 0, 2), DATA3D(2, 0, 2), DATA3D(3, 0, 2));
                                    yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2));
                                    yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2));
                                    yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 3, 2), DATA3D(1, 3, 2), DATA3D(2, 3, 2), DATA3D(3, 3, 2));
                                 }
                                 ymtmp[2] = POLINT_MACRO_CUBIC_R2(yntmp[0],yntmp[1],yntmp[2],yntmp[3]);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_CUBIC(DATA3D(0, 0, 3), DATA3D(1, 0, 3), DATA3D(2, 0, 3), DATA3D(3, 0, 3), Rx, px);
                                    yntmp[1] = POLINT_MACRO_CUBIC(DATA3D(0, 1, 3), DATA3D(1, 1, 3), DATA3D(2, 1, 3), DATA3D(3, 1, 3), Rx, px);
                                    yntmp[2] = POLINT_MACRO_CUBIC(DATA3D(0, 2, 3), DATA3D(1, 2, 3), DATA3D(2, 2, 3), DATA3D(3, 2, 3), Rx, px);
                                    yntmp[3] = POLINT_MACRO_CUBIC(DATA3D(0, 3, 3), DATA3D(1, 3, 3), DATA3D(2, 3, 3), DATA3D(3, 3, 3), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 0, 3), DATA3D(1, 0, 3), DATA3D(2, 0, 3), DATA3D(3, 0, 3));
                                    yntmp[1] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 1, 3), DATA3D(1, 1, 3), DATA3D(2, 1, 3), DATA3D(3, 1, 3));
                                    yntmp[2] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 2, 3), DATA3D(1, 2, 3), DATA3D(2, 2, 3), DATA3D(3, 2, 3));
                                    yntmp[3] = POLINT_MACRO_CUBIC_R2(DATA3D(0, 3, 3), DATA3D(1, 3, 3), DATA3D(2, 3, 3), DATA3D(3, 3, 3));
                                 }
                                 ymtmp[3] = POLINT_MACRO_CUBIC_R2(yntmp[0],yntmp[1],yntmp[2],yntmp[3]);
                              }
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC(ymtmp[0],ymtmp[1],ymtmp[2],ymtmp[3], Rz, pz);
                              }
                              else {   
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) = POLINT_MACRO_CUBIC_R2(ymtmp[0],ymtmp[1],ymtmp[2],ymtmp[3]);
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      } else {
         TBOX_ERROR("LagrangianPolynomicRefine error...\n"
            << "dimension > 3 or < 2 not supported." << endl);

      }
   }
}

void LagrangianPolynomicRefine::refine_quintic(
   hier::Patch& fine,
   const hier::Patch& coarse,
   const int dst_component,
   const int src_component,
   const hier::Box& fine_box,
   const hier::IntVector& ratio) const
{
   std::shared_ptr<pdat::NodeData<double> > cdata(
      SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(
         coarse.getPatchData(src_component)));
   std::shared_ptr<pdat::NodeData<double> > fdata(
      SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(
         fine.getPatchData(dst_component)));

   TBOX_ASSERT(cdata);
   TBOX_ASSERT(fdata);
   TBOX_ASSERT(cdata->getDepth() == fdata->getDepth());

   const hier::Index cilo = cdata->getGhostBox().lower();
   const hier::Index cihi = cdata->getGhostBox().upper();
   const hier::Index filo = fdata->getGhostBox().lower();
   const hier::Index fihi = fdata->getGhostBox().upper();

   const hier::Index il = fdata->getBox().lower();
   const hier::Index ih = fdata->getBox().upper();

   const hier::Index ifirstf = fine_box.lower();
   const hier::Index ilastf = fine_box.upper();

   int Rx = ratio[0];
   int Ry = ratio[1];

   std::vector<double> ya;
   int kbase, jbase, ibase; 

   double ymtmp[6];
   double yntmp[6];

   for (int d = 0; d < fdata->getDepth(); ++d) {

      double* fieldf = fdata->getPointer(d);
      double* fieldc = cdata->getPointer(d);

      if (fine.getDim() == tbox::Dimension(2)) {
         int ilast = fihi(0)-filo(0) + 2;

         int ilastC = cihi(0)-cilo(0) + 2;

         double ratio0f = (double) ratio[0];
         double ratio1f = (double) ratio[1];

         int nrows = (floor(ilastf[0] / ratio0f) + d_half_stencil) - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1) + 1;
         //Initialization of coarser value array
         for(int j = floor(ifirstf[1] / ratio1f) - d_half_stencil + 1; j <= floor(ilastf[1] / ratio1f) + d_half_stencil; j++) {
            for(int i = floor(ifirstf[0] / ratio0f) - d_half_stencil + 1; i <= floor(ilastf[0] / ratio0f) + d_half_stencil; i++) {
               ya.push_back(vector2DC(fieldc, i - cilo[0], j - cilo[1]));
            }
         }
         double *yat = &ya[0];
         //Iterate over fine nodes and interpolate
         for(int j = ifirstf[1]; j <= ilastf[1]; j++) {
            jbase = floor(j / ratio1f) - d_half_stencil + 1 - (floor(ifirstf[1] / ratio1f) - d_half_stencil + 1);
            int py = abs(j % Ry);
            bool y_alingment = py == 0;
            for(int i = ifirstf[0]; i <= ilastf[0]; i++) {
               ibase = floor(i / ratio0f) - d_half_stencil + 1 - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1);
               int px = abs(i % Rx);
               bool x_alingment = px == 0;
               if (d_refine_boundary || i < il[0] || i > ih[0]+1 || j < il[1] || j > ih[1] + 1) {
                  if (x_alingment) {
                     if (y_alingment) {
                        vector2D(fieldf, i - filo[0], j - filo[1]) = DATA2D(2, 2);
                     }
                     else {
                        if (Ry > 2) {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_QUINTIC(DATA2D(2, 0), DATA2D(2, 1), DATA2D(2, 2), DATA2D(2, 3), DATA2D(2, 4), DATA2D(2, 5), Ry, py);
                        }
                        else {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_QUINTIC_R2(DATA2D(2, 0), DATA2D(2, 1), DATA2D(2, 2), DATA2D(2, 3), DATA2D(2, 4), DATA2D(2, 5));
                        }
                     }
                  }
                  else {
                     if (y_alingment) {
                        if (Rx > 2) {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_QUINTIC(DATA2D(0, 2), DATA2D(1, 2), DATA2D(2, 2), DATA2D(3, 2), DATA2D(4, 2), DATA2D(5, 2), Rx, px);
                        }
                        else {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_QUINTIC_R2(DATA2D(0, 2), DATA2D(1, 2), DATA2D(2, 2), DATA2D(3, 2), DATA2D(4, 2), DATA2D(5, 2));
                        }
                     }
                     else {
                        if (Rx > 2) {
                           ymtmp[0] = POLINT_MACRO_QUINTIC(DATA2D(0, 0), DATA2D(1, 0), DATA2D(2, 0), DATA2D(3, 0), DATA2D(4, 0), DATA2D(5, 0), Rx, px);
                           ymtmp[1] = POLINT_MACRO_QUINTIC(DATA2D(0, 1), DATA2D(1, 1), DATA2D(2, 1), DATA2D(3, 1), DATA2D(4, 1), DATA2D(5, 1), Rx, px);
                           ymtmp[2] = POLINT_MACRO_QUINTIC(DATA2D(0, 2), DATA2D(1, 2), DATA2D(2, 2), DATA2D(3, 2), DATA2D(4, 2), DATA2D(5, 2), Rx, px);
                           ymtmp[3] = POLINT_MACRO_QUINTIC(DATA2D(0, 3), DATA2D(1, 3), DATA2D(2, 3), DATA2D(3, 3), DATA2D(4, 3), DATA2D(5, 3), Rx, px);
                           ymtmp[4] = POLINT_MACRO_QUINTIC(DATA2D(0, 4), DATA2D(1, 4), DATA2D(2, 4), DATA2D(3, 4), DATA2D(4, 4), DATA2D(5, 4), Rx, px);
                           ymtmp[5] = POLINT_MACRO_QUINTIC(DATA2D(0, 5), DATA2D(1, 5), DATA2D(2, 5), DATA2D(3, 5), DATA2D(4, 5), DATA2D(5, 5), Rx, px);
                        }
                        else {
                           ymtmp[0] = POLINT_MACRO_QUINTIC_R2(DATA2D(0, 0), DATA2D(1, 0), DATA2D(2, 0), DATA2D(3, 0), DATA2D(4, 0), DATA2D(5, 0));
                           ymtmp[1] = POLINT_MACRO_QUINTIC_R2(DATA2D(0, 1), DATA2D(1, 1), DATA2D(2, 1), DATA2D(3, 1), DATA2D(4, 1), DATA2D(5, 1));
                           ymtmp[2] = POLINT_MACRO_QUINTIC_R2(DATA2D(0, 2), DATA2D(1, 2), DATA2D(2, 2), DATA2D(3, 2), DATA2D(4, 2), DATA2D(5, 2));
                           ymtmp[3] = POLINT_MACRO_QUINTIC_R2(DATA2D(0, 3), DATA2D(1, 3), DATA2D(2, 3), DATA2D(3, 3), DATA2D(4, 3), DATA2D(5, 3));
                           ymtmp[4] = POLINT_MACRO_QUINTIC_R2(DATA2D(0, 4), DATA2D(1, 4), DATA2D(2, 4), DATA2D(3, 4), DATA2D(4, 4), DATA2D(5, 4));
                           ymtmp[5] = POLINT_MACRO_QUINTIC_R2(DATA2D(0, 5), DATA2D(1, 5), DATA2D(2, 5), DATA2D(3, 5), DATA2D(4, 5), DATA2D(5, 5));
                        }
                        if (Ry > 2) {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_QUINTIC(ymtmp[0], ymtmp[1], ymtmp[2], ymtmp[3], ymtmp[4], ymtmp[5], Ry, py);   
                        }
                        else {
                           vector2D(fieldf, i - filo[0], j - filo[1]) = POLINT_MACRO_QUINTIC_R2(ymtmp[0], ymtmp[1], ymtmp[2], ymtmp[3], ymtmp[4], ymtmp[5]);
                        }
                     }
                  }
               }
            }
         }
      } else if (fine.getDim() == tbox::Dimension(3)) {
         int ilast = fihi(0)-filo(0) + 2;
         int jlast = fihi(1)-filo(1) + 2;

         int ilastC = cihi(0)-cilo(0) + 2;
         int jlastC = cihi(1)-cilo(1) + 2;

         double ratio0f = (double) ratio[0];
         double ratio1f = (double) ratio[1];
         double ratio2f = (double) ratio[2];
         int Rz = ratio[2];

         int nrows = (floor(ilastf[0] / ratio0f) + d_half_stencil) - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1) + 1;
         int ncolumns = (floor(ilastf[1] / ratio1f) + d_half_stencil) - (floor(ifirstf[1] / ratio1f) - d_half_stencil + 1) + 1;
         //Initialization of coarser value array
         for(int k = floor(ifirstf[2] / ratio2f) - d_half_stencil + 1; k <= floor(ilastf[2] / ratio2f) + d_half_stencil; k++) {
            for(int j = floor(ifirstf[1] / ratio1f) - d_half_stencil + 1; j <= floor(ilastf[1] / ratio1f) + d_half_stencil; j++) {
               for(int i = floor(ifirstf[0] / ratio0f) - d_half_stencil + 1; i <= floor(ilastf[0] / ratio0f) + d_half_stencil; i++) {
                  ya.push_back(vector3DC(fieldc, i - cilo[0], j - cilo[1], k - cilo[2]));
               }
            }
         }

         double *yat = &ya[0];
         //Iterate over fine nodes and interpolate
         for(int k = ifirstf[2]; k <= ilastf[2]; k++) {
            kbase = floor(k / ratio2f) - d_half_stencil + 1 - (floor(ifirstf[2] / ratio2f) - d_half_stencil + 1);
            int pz = abs(k % Rz);
            bool z_alingment = pz == 0;
            for(int j = ifirstf[1]; j <= ilastf[1]; j++) {
               jbase = floor(j / ratio1f) - d_half_stencil + 1 - (floor(ifirstf[1] / ratio1f) - d_half_stencil + 1);
               int py = abs(j % Ry);
               bool y_alingment = py == 0;
               for(int i = ifirstf[0]; i <= ilastf[0]; i++) {
                  ibase = floor(i / ratio0f) - d_half_stencil + 1 - (floor(ifirstf[0] / ratio0f) - d_half_stencil + 1);
                  int px = abs(i % Rx);
                  bool x_alingment = px == 0;
                  if (d_refine_boundary || i < il[0] || i > ih[0]+1 || j < il[1] || j > ih[1] + 1 || k < il[2] || k > ih[2] + 1) {
                     if (x_alingment) {
                        if (y_alingment) {
                           if (z_alingment) {
                              vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  DATA3D(2, 2, 2);
                           }
                           else {
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC(DATA3D(2, 2, 0), DATA3D(2, 2, 1), DATA3D(2, 2, 2), DATA3D(2, 2, 3), DATA3D(2, 2, 4), DATA3D(2, 2, 5), Rz, pz);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC_R2(DATA3D(2, 2, 0), DATA3D(2, 2, 1), DATA3D(2, 2, 2), DATA3D(2, 2, 3), DATA3D(2, 2, 4), DATA3D(2, 2, 5));                               
                              }                           
                           }
                        }
                        else {
                           if (z_alingment) {
                              if (Ry > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC(DATA3D(2, 0, 2), DATA3D(2, 1, 2), DATA3D(2, 2, 2), DATA3D(2, 3, 2), DATA3D(2, 4, 2), DATA3D(2, 5, 2), Ry, py);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC_R2(DATA3D(2, 0, 2), DATA3D(2, 1, 2), DATA3D(2, 2, 2), DATA3D(2, 3, 2), DATA3D(2, 4, 2), DATA3D(2, 5, 2));                               
                              }
                           }
                           else {
                              if (Ry > 2) {
                                 yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(2, 0, 0), DATA3D(2, 1, 0), DATA3D(2, 2, 0), DATA3D(2, 3, 0), DATA3D(2, 4, 0), DATA3D(2, 5, 0), Ry, py);
                                 yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(2, 0, 1), DATA3D(2, 1, 1), DATA3D(2, 2, 1), DATA3D(2, 3, 1), DATA3D(2, 4, 1), DATA3D(2, 5, 1), Ry, py);
                                 yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(2, 0, 2), DATA3D(2, 1, 2), DATA3D(2, 2, 2), DATA3D(2, 3, 2), DATA3D(2, 4, 2), DATA3D(2, 5, 2), Ry, py);
                                 yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(2, 0, 3), DATA3D(2, 1, 3), DATA3D(2, 2, 3), DATA3D(2, 3, 3), DATA3D(2, 4, 3), DATA3D(2, 5, 3), Ry, py);
                                 yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(2, 0, 4), DATA3D(2, 1, 4), DATA3D(2, 2, 4), DATA3D(2, 3, 4), DATA3D(2, 4, 4), DATA3D(2, 5, 4), Ry, py);
                                 yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(2, 0, 5), DATA3D(2, 1, 5), DATA3D(2, 2, 5), DATA3D(2, 3, 5), DATA3D(2, 4, 5), DATA3D(2, 5, 5), Ry, py);
                              }
                              else {
                                 yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(2, 0, 0), DATA3D(2, 1, 0), DATA3D(2, 2, 0), DATA3D(2, 3, 0), DATA3D(2, 4, 0), DATA3D(2, 5, 0));
                                 yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(2, 0, 1), DATA3D(2, 1, 1), DATA3D(2, 2, 1), DATA3D(2, 3, 1), DATA3D(2, 4, 1), DATA3D(2, 5, 1));
                                 yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(2, 0, 2), DATA3D(2, 1, 2), DATA3D(2, 2, 2), DATA3D(2, 3, 2), DATA3D(2, 4, 2), DATA3D(2, 5, 2));
                                 yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(2, 0, 3), DATA3D(2, 1, 3), DATA3D(2, 2, 3), DATA3D(2, 3, 3), DATA3D(2, 4, 3), DATA3D(2, 5, 3));
                                 yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(2, 0, 4), DATA3D(2, 1, 4), DATA3D(2, 2, 4), DATA3D(2, 3, 4), DATA3D(2, 4, 4), DATA3D(2, 5, 4));
                                 yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(2, 0, 5), DATA3D(2, 1, 5), DATA3D(2, 2, 5), DATA3D(2, 3, 5), DATA3D(2, 4, 5), DATA3D(2, 5, 5));
                              }
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5], Rz, pz);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC_R2(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5]);
                              }
                           }
                        }
                     }
                     else {
                        if (y_alingment) {
                           if (z_alingment) {
                              if (Rx > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), DATA3D(4, 2, 2), DATA3D(5, 2, 2), Rx, px);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), DATA3D(4, 2, 2), DATA3D(5, 2, 2));                               
                              }
                           }
                           else {
                              if (Rx > 2) {
                                 yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 0), DATA3D(1, 2, 0), DATA3D(2, 2, 0), DATA3D(3, 2, 0), DATA3D(4, 2, 0), DATA3D(5, 2, 0), Rx, px);
                                 yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1), DATA3D(4, 2, 1), DATA3D(5, 2, 1), Rx, px);
                                 yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), DATA3D(4, 2, 2), DATA3D(5, 2, 2), Rx, px);
                                 yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 3), DATA3D(1, 2, 3), DATA3D(2, 2, 3), DATA3D(3, 2, 3), DATA3D(4, 2, 3), DATA3D(5, 2, 3), Rx, px);
                                 yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 4), DATA3D(1, 2, 4), DATA3D(2, 2, 4), DATA3D(3, 2, 4), DATA3D(4, 2, 4), DATA3D(5, 2, 4), Rx, px);
                                 yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 5), DATA3D(1, 2, 5), DATA3D(2, 2, 5), DATA3D(3, 2, 5), DATA3D(4, 2, 5), DATA3D(5, 2, 5), Rx, px);
                              }
                              else {
                                 yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 0), DATA3D(1, 2, 0), DATA3D(2, 2, 0), DATA3D(3, 2, 0), DATA3D(4, 2, 0), DATA3D(5, 2, 0));
                                 yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1), DATA3D(4, 2, 1), DATA3D(5, 2, 1));
                                 yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), DATA3D(4, 2, 2), DATA3D(5, 2, 2));
                                 yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 3), DATA3D(1, 2, 3), DATA3D(2, 2, 3), DATA3D(3, 2, 3), DATA3D(4, 2, 3), DATA3D(5, 2, 3));
                                 yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 4), DATA3D(1, 2, 4), DATA3D(2, 2, 4), DATA3D(3, 2, 4), DATA3D(4, 2, 4), DATA3D(5, 2, 4));
                                 yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 5), DATA3D(1, 2, 5), DATA3D(2, 2, 5), DATA3D(3, 2, 5), DATA3D(4, 2, 5), DATA3D(5, 2, 5));
                              }
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5], Rz, pz);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC_R2(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5]);
                              }
                           }
                        }
                        else {
                           if (z_alingment) {
                              if (Rx > 2) {
                                 yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 2), DATA3D(1, 0, 2), DATA3D(2, 0, 2), DATA3D(3, 0, 2), DATA3D(4, 0, 2), DATA3D(5, 0, 2), Rx, px);
                                 yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2), DATA3D(4, 1, 2), DATA3D(5, 1, 2), Rx, px);
                                 yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), DATA3D(4, 2, 2), DATA3D(5, 2, 2), Rx, px);
                                 yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 2), DATA3D(1, 3, 2), DATA3D(2, 3, 2), DATA3D(3, 3, 2), DATA3D(4, 3, 2), DATA3D(5, 3, 2), Rx, px);
                                 yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 2), DATA3D(1, 4, 2), DATA3D(2, 4, 2), DATA3D(3, 4, 2), DATA3D(4, 4, 2), DATA3D(5, 4, 2), Rx, px);
                                 yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 2), DATA3D(1, 5, 2), DATA3D(2, 5, 2), DATA3D(3, 5, 2), DATA3D(4, 5, 2), DATA3D(5, 5, 2), Rx, px);
                              }
                              else {
                                 yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 2), DATA3D(1, 0, 2), DATA3D(2, 0, 2), DATA3D(3, 0, 2), DATA3D(4, 0, 2), DATA3D(5, 0, 2));
                                 yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2), DATA3D(4, 1, 2), DATA3D(5, 1, 2));
                                 yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), DATA3D(4, 2, 2), DATA3D(5, 2, 2));
                                 yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 2), DATA3D(1, 3, 2), DATA3D(2, 3, 2), DATA3D(3, 3, 2), DATA3D(4, 3, 2), DATA3D(5, 3, 2));
                                 yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 2), DATA3D(1, 4, 2), DATA3D(2, 4, 2), DATA3D(3, 4, 2), DATA3D(4, 4, 2), DATA3D(5, 4, 2));
                                 yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 2), DATA3D(1, 5, 2), DATA3D(2, 5, 2), DATA3D(3, 5, 2), DATA3D(4, 5, 2), DATA3D(5, 5, 2));
                              }
                              if (Ry > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5], Ry, py);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC_R2(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5]);
                              }
                           }
                           else {
                              if (Ry > 2) {
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 0), DATA3D(1, 0, 0), DATA3D(2, 0, 0), DATA3D(3, 0, 0), DATA3D(4, 0, 0), DATA3D(5, 0, 0), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 0), DATA3D(1, 1, 0), DATA3D(2, 1, 0), DATA3D(3, 1, 0), DATA3D(4, 1, 0), DATA3D(5, 1, 0), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 0), DATA3D(1, 2, 0), DATA3D(2, 2, 0), DATA3D(3, 2, 0), DATA3D(4, 2, 0), DATA3D(5, 2, 0), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 0), DATA3D(1, 3, 0), DATA3D(2, 3, 0), DATA3D(3, 3, 0), DATA3D(4, 3, 0), DATA3D(5, 3, 0), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 0), DATA3D(1, 4, 0), DATA3D(2, 4, 0), DATA3D(3, 4, 0), DATA3D(4, 4, 0), DATA3D(5, 4, 0), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 0), DATA3D(1, 5, 0), DATA3D(2, 5, 0), DATA3D(3, 5, 0), DATA3D(4, 5, 0), DATA3D(5, 5, 0), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 0), DATA3D(1, 0, 0), DATA3D(2, 0, 0), DATA3D(3, 0, 0), DATA3D(4, 0, 0), DATA3D(5, 0, 0));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 0), DATA3D(1, 1, 0), DATA3D(2, 1, 0), DATA3D(3, 1, 0), DATA3D(4, 1, 0), DATA3D(5, 1, 0));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 0), DATA3D(1, 2, 0), DATA3D(2, 2, 0), DATA3D(3, 2, 0), DATA3D(4, 2, 0), DATA3D(5, 2, 0));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 0), DATA3D(1, 3, 0), DATA3D(2, 3, 0), DATA3D(3, 3, 0), DATA3D(4, 3, 0), DATA3D(5, 3, 0));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 0), DATA3D(1, 4, 0), DATA3D(2, 4, 0), DATA3D(3, 4, 0), DATA3D(4, 4, 0), DATA3D(5, 4, 0));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 0), DATA3D(1, 5, 0), DATA3D(2, 5, 0), DATA3D(3, 5, 0), DATA3D(4, 5, 0), DATA3D(5, 5, 0));
                                 }
                                 ymtmp[0] =  POLINT_MACRO_QUINTIC(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5], Ry, py);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 1), DATA3D(1, 0, 1), DATA3D(2, 0, 1), DATA3D(3, 0, 1), DATA3D(4, 0, 1), DATA3D(5, 0, 1), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1), DATA3D(4, 1, 1), DATA3D(5, 1, 1), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1), DATA3D(4, 2, 1), DATA3D(5, 2, 1), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 1), DATA3D(1, 3, 1), DATA3D(2, 3, 1), DATA3D(3, 3, 1), DATA3D(4, 3, 1), DATA3D(5, 3, 1), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 1), DATA3D(1, 4, 1), DATA3D(2, 4, 1), DATA3D(3, 4, 1), DATA3D(4, 4, 1), DATA3D(5, 4, 1), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 1), DATA3D(1, 5, 1), DATA3D(2, 5, 1), DATA3D(3, 5, 1), DATA3D(4, 5, 1), DATA3D(5, 5, 1), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 1), DATA3D(1, 0, 1), DATA3D(2, 0, 1), DATA3D(3, 0, 1), DATA3D(4, 0, 1), DATA3D(5, 0, 1));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1), DATA3D(4, 1, 1), DATA3D(5, 1, 1));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1), DATA3D(4, 2, 1), DATA3D(5, 2, 1));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 1), DATA3D(1, 3, 1), DATA3D(2, 3, 1), DATA3D(3, 3, 1), DATA3D(4, 3, 1), DATA3D(5, 3, 1));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 1), DATA3D(1, 4, 1), DATA3D(2, 4, 1), DATA3D(3, 4, 1), DATA3D(4, 4, 1), DATA3D(5, 4, 1));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 1), DATA3D(1, 5, 1), DATA3D(2, 5, 1), DATA3D(3, 5, 1), DATA3D(4, 5, 1), DATA3D(5, 5, 1));
                                 }
                                 ymtmp[1] =  POLINT_MACRO_QUINTIC(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5], Ry, py);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 2), DATA3D(1, 0, 2), DATA3D(2, 0, 2), DATA3D(3, 0, 2), DATA3D(4, 0, 2), DATA3D(5, 0, 2), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2), DATA3D(4, 1, 2), DATA3D(5, 1, 2), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), DATA3D(4, 2, 2), DATA3D(5, 2, 2), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 2), DATA3D(1, 3, 2), DATA3D(2, 3, 2), DATA3D(3, 3, 2), DATA3D(4, 3, 2), DATA3D(5, 3, 2), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 2), DATA3D(1, 4, 2), DATA3D(2, 4, 2), DATA3D(3, 4, 2), DATA3D(4, 4, 2), DATA3D(5, 4, 2), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 2), DATA3D(1, 5, 2), DATA3D(2, 5, 2), DATA3D(3, 5, 2), DATA3D(4, 5, 2), DATA3D(5, 5, 2), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 2), DATA3D(1, 0, 2), DATA3D(2, 0, 2), DATA3D(3, 0, 2), DATA3D(4, 0, 2), DATA3D(5, 0, 2));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2), DATA3D(4, 1, 2), DATA3D(5, 1, 2));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), DATA3D(4, 2, 2), DATA3D(5, 2, 2));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 2), DATA3D(1, 3, 2), DATA3D(2, 3, 2), DATA3D(3, 3, 2), DATA3D(4, 3, 2), DATA3D(5, 3, 2));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 2), DATA3D(1, 4, 2), DATA3D(2, 4, 2), DATA3D(3, 4, 2), DATA3D(4, 4, 2), DATA3D(5, 4, 2));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 2), DATA3D(1, 5, 2), DATA3D(2, 5, 2), DATA3D(3, 5, 2), DATA3D(4, 5, 2), DATA3D(5, 5, 2));
                                 }
                                 ymtmp[2] =  POLINT_MACRO_QUINTIC(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5], Ry, py);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 3), DATA3D(1, 0, 3), DATA3D(2, 0, 3), DATA3D(3, 0, 3), DATA3D(4, 0, 3), DATA3D(5, 0, 3), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 3), DATA3D(1, 1, 3), DATA3D(2, 1, 3), DATA3D(3, 1, 3), DATA3D(4, 1, 3), DATA3D(5, 1, 3), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 3), DATA3D(1, 2, 3), DATA3D(2, 2, 3), DATA3D(3, 2, 3), DATA3D(4, 2, 3), DATA3D(5, 2, 3), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 3), DATA3D(1, 3, 3), DATA3D(2, 3, 3), DATA3D(3, 3, 3), DATA3D(4, 3, 3), DATA3D(5, 3, 3), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 3), DATA3D(1, 4, 3), DATA3D(2, 4, 3), DATA3D(3, 4, 3), DATA3D(4, 4, 3), DATA3D(5, 4, 3), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 3), DATA3D(1, 5, 3), DATA3D(2, 5, 3), DATA3D(3, 5, 3), DATA3D(4, 5, 3), DATA3D(5, 5, 3), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 3), DATA3D(1, 0, 3), DATA3D(2, 0, 3), DATA3D(3, 0, 3), DATA3D(4, 0, 3), DATA3D(5, 0, 3));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 3), DATA3D(1, 1, 3), DATA3D(2, 1, 3), DATA3D(3, 1, 3), DATA3D(4, 1, 3), DATA3D(5, 1, 3));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 3), DATA3D(1, 2, 3), DATA3D(2, 2, 3), DATA3D(3, 2, 3), DATA3D(4, 2, 3), DATA3D(5, 2, 3));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 3), DATA3D(1, 3, 3), DATA3D(2, 3, 3), DATA3D(3, 3, 3), DATA3D(4, 3, 3), DATA3D(5, 3, 3));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 3), DATA3D(1, 4, 3), DATA3D(2, 4, 3), DATA3D(3, 4, 3), DATA3D(4, 4, 3), DATA3D(5, 4, 3));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 3), DATA3D(1, 5, 3), DATA3D(2, 5, 3), DATA3D(3, 5, 3), DATA3D(4, 5, 3), DATA3D(5, 5, 3));
                                 }
                                 ymtmp[3] =  POLINT_MACRO_QUINTIC(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5], Ry, py);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 4), DATA3D(1, 0, 4), DATA3D(2, 0, 4), DATA3D(3, 0, 4), DATA3D(4, 0, 4), DATA3D(5, 0, 4), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 4), DATA3D(1, 1, 4), DATA3D(2, 1, 4), DATA3D(3, 1, 4), DATA3D(4, 1, 4), DATA3D(5, 1, 4), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 4), DATA3D(1, 2, 4), DATA3D(2, 2, 4), DATA3D(3, 2, 4), DATA3D(4, 2, 4), DATA3D(5, 2, 4), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 4), DATA3D(1, 3, 4), DATA3D(2, 3, 4), DATA3D(3, 3, 4), DATA3D(4, 3, 4), DATA3D(5, 3, 4), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 4), DATA3D(1, 4, 4), DATA3D(2, 4, 4), DATA3D(3, 4, 4), DATA3D(4, 4, 4), DATA3D(5, 4, 4), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 4), DATA3D(1, 5, 4), DATA3D(2, 5, 4), DATA3D(3, 5, 4), DATA3D(4, 5, 4), DATA3D(5, 5, 4), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 4), DATA3D(1, 0, 4), DATA3D(2, 0, 4), DATA3D(3, 0, 4), DATA3D(4, 0, 4), DATA3D(5, 0, 4));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 4), DATA3D(1, 1, 4), DATA3D(2, 1, 4), DATA3D(3, 1, 4), DATA3D(4, 1, 4), DATA3D(5, 1, 4));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 4), DATA3D(1, 2, 4), DATA3D(2, 2, 4), DATA3D(3, 2, 4), DATA3D(4, 2, 4), DATA3D(5, 2, 4));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 4), DATA3D(1, 3, 4), DATA3D(2, 3, 4), DATA3D(3, 3, 4), DATA3D(4, 3, 4), DATA3D(5, 3, 4));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 4), DATA3D(1, 4, 4), DATA3D(2, 4, 4), DATA3D(3, 4, 4), DATA3D(4, 4, 4), DATA3D(5, 4, 4));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 4), DATA3D(1, 5, 4), DATA3D(2, 5, 4), DATA3D(3, 5, 4), DATA3D(4, 5, 4), DATA3D(5, 5, 4));
                                 }
                                 ymtmp[4] =  POLINT_MACRO_QUINTIC(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5], Ry, py);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 5), DATA3D(1, 0, 5), DATA3D(2, 0, 5), DATA3D(3, 0, 5), DATA3D(4, 0, 5), DATA3D(5, 0, 5), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 5), DATA3D(1, 1, 5), DATA3D(2, 1, 5), DATA3D(3, 1, 5), DATA3D(4, 1, 5), DATA3D(5, 1, 5), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 5), DATA3D(1, 2, 5), DATA3D(2, 2, 5), DATA3D(3, 2, 5), DATA3D(4, 2, 5), DATA3D(5, 2, 5), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 5), DATA3D(1, 3, 5), DATA3D(2, 3, 5), DATA3D(3, 3, 5), DATA3D(4, 3, 5), DATA3D(5, 3, 5), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 5), DATA3D(1, 4, 5), DATA3D(2, 4, 5), DATA3D(3, 4, 5), DATA3D(4, 4, 5), DATA3D(5, 4, 5), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 5), DATA3D(1, 5, 5), DATA3D(2, 5, 5), DATA3D(3, 5, 5), DATA3D(4, 5, 5), DATA3D(5, 5, 5), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 5), DATA3D(1, 0, 5), DATA3D(2, 0, 5), DATA3D(3, 0, 5), DATA3D(4, 0, 5), DATA3D(5, 0, 5));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 5), DATA3D(1, 1, 5), DATA3D(2, 1, 5), DATA3D(3, 1, 5), DATA3D(4, 1, 5), DATA3D(5, 1, 5));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 5), DATA3D(1, 2, 5), DATA3D(2, 2, 5), DATA3D(3, 2, 5), DATA3D(4, 2, 5), DATA3D(5, 2, 5));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 5), DATA3D(1, 3, 5), DATA3D(2, 3, 5), DATA3D(3, 3, 5), DATA3D(4, 3, 5), DATA3D(5, 3, 5));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 5), DATA3D(1, 4, 5), DATA3D(2, 4, 5), DATA3D(3, 4, 5), DATA3D(4, 4, 5), DATA3D(5, 4, 5));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 5), DATA3D(1, 5, 5), DATA3D(2, 5, 5), DATA3D(3, 5, 5), DATA3D(4, 5, 5), DATA3D(5, 5, 5));
                                 }
                                 ymtmp[5] =  POLINT_MACRO_QUINTIC(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5], Ry, py);
                              }
                              else {
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 0), DATA3D(1, 0, 0), DATA3D(2, 0, 0), DATA3D(3, 0, 0), DATA3D(4, 0, 0), DATA3D(5, 0, 0), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 0), DATA3D(1, 1, 0), DATA3D(2, 1, 0), DATA3D(3, 1, 0), DATA3D(4, 1, 0), DATA3D(5, 1, 0), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 0), DATA3D(1, 2, 0), DATA3D(2, 2, 0), DATA3D(3, 2, 0), DATA3D(4, 2, 0), DATA3D(5, 2, 0), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 0), DATA3D(1, 3, 0), DATA3D(2, 3, 0), DATA3D(3, 3, 0), DATA3D(4, 3, 0), DATA3D(5, 3, 0), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 0), DATA3D(1, 4, 0), DATA3D(2, 4, 0), DATA3D(3, 4, 0), DATA3D(4, 4, 0), DATA3D(5, 4, 0), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 0), DATA3D(1, 5, 0), DATA3D(2, 5, 0), DATA3D(3, 5, 0), DATA3D(4, 5, 0), DATA3D(5, 5, 0), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 0), DATA3D(1, 0, 0), DATA3D(2, 0, 0), DATA3D(3, 0, 0), DATA3D(4, 0, 0), DATA3D(5, 0, 0));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 0), DATA3D(1, 1, 0), DATA3D(2, 1, 0), DATA3D(3, 1, 0), DATA3D(4, 1, 0), DATA3D(5, 1, 0));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 0), DATA3D(1, 2, 0), DATA3D(2, 2, 0), DATA3D(3, 2, 0), DATA3D(4, 2, 0), DATA3D(5, 2, 0));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 0), DATA3D(1, 3, 0), DATA3D(2, 3, 0), DATA3D(3, 3, 0), DATA3D(4, 3, 0), DATA3D(5, 3, 0));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 0), DATA3D(1, 4, 0), DATA3D(2, 4, 0), DATA3D(3, 4, 0), DATA3D(4, 4, 0), DATA3D(5, 4, 0));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 0), DATA3D(1, 5, 0), DATA3D(2, 5, 0), DATA3D(3, 5, 0), DATA3D(4, 5, 0), DATA3D(5, 5, 0));
                                 }
                                 ymtmp[0] =  POLINT_MACRO_QUINTIC_R2(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5]);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 1), DATA3D(1, 0, 1), DATA3D(2, 0, 1), DATA3D(3, 0, 1), DATA3D(4, 0, 1), DATA3D(5, 0, 1), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1), DATA3D(4, 1, 1), DATA3D(5, 1, 1), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1), DATA3D(4, 2, 1), DATA3D(5, 2, 1), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 1), DATA3D(1, 3, 1), DATA3D(2, 3, 1), DATA3D(3, 3, 1), DATA3D(4, 3, 1), DATA3D(5, 3, 1), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 1), DATA3D(1, 4, 1), DATA3D(2, 4, 1), DATA3D(3, 4, 1), DATA3D(4, 4, 1), DATA3D(5, 4, 1), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 1), DATA3D(1, 5, 1), DATA3D(2, 5, 1), DATA3D(3, 5, 1), DATA3D(4, 5, 1), DATA3D(5, 5, 1), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 1), DATA3D(1, 0, 1), DATA3D(2, 0, 1), DATA3D(3, 0, 1), DATA3D(4, 0, 1), DATA3D(5, 0, 1));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 1), DATA3D(1, 1, 1), DATA3D(2, 1, 1), DATA3D(3, 1, 1), DATA3D(4, 1, 1), DATA3D(5, 1, 1));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 1), DATA3D(1, 2, 1), DATA3D(2, 2, 1), DATA3D(3, 2, 1), DATA3D(4, 2, 1), DATA3D(5, 2, 1));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 1), DATA3D(1, 3, 1), DATA3D(2, 3, 1), DATA3D(3, 3, 1), DATA3D(4, 3, 1), DATA3D(5, 3, 1));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 1), DATA3D(1, 4, 1), DATA3D(2, 4, 1), DATA3D(3, 4, 1), DATA3D(4, 4, 1), DATA3D(5, 4, 1));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 1), DATA3D(1, 5, 1), DATA3D(2, 5, 1), DATA3D(3, 5, 1), DATA3D(4, 5, 1), DATA3D(5, 5, 1));
                                 }
                                 ymtmp[1] =  POLINT_MACRO_QUINTIC_R2(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5]);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 2), DATA3D(1, 0, 2), DATA3D(2, 0, 2), DATA3D(3, 0, 2), DATA3D(4, 0, 2), DATA3D(5, 0, 2), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2), DATA3D(4, 1, 2), DATA3D(5, 1, 2), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), DATA3D(4, 2, 2), DATA3D(5, 2, 2), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 2), DATA3D(1, 3, 2), DATA3D(2, 3, 2), DATA3D(3, 3, 2), DATA3D(4, 3, 2), DATA3D(5, 3, 2), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 2), DATA3D(1, 4, 2), DATA3D(2, 4, 2), DATA3D(3, 4, 2), DATA3D(4, 4, 2), DATA3D(5, 4, 2), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 2), DATA3D(1, 5, 2), DATA3D(2, 5, 2), DATA3D(3, 5, 2), DATA3D(4, 5, 2), DATA3D(5, 5, 2), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 2), DATA3D(1, 0, 2), DATA3D(2, 0, 2), DATA3D(3, 0, 2), DATA3D(4, 0, 2), DATA3D(5, 0, 2));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 2), DATA3D(1, 1, 2), DATA3D(2, 1, 2), DATA3D(3, 1, 2), DATA3D(4, 1, 2), DATA3D(5, 1, 2));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 2), DATA3D(1, 2, 2), DATA3D(2, 2, 2), DATA3D(3, 2, 2), DATA3D(4, 2, 2), DATA3D(5, 2, 2));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 2), DATA3D(1, 3, 2), DATA3D(2, 3, 2), DATA3D(3, 3, 2), DATA3D(4, 3, 2), DATA3D(5, 3, 2));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 2), DATA3D(1, 4, 2), DATA3D(2, 4, 2), DATA3D(3, 4, 2), DATA3D(4, 4, 2), DATA3D(5, 4, 2));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 2), DATA3D(1, 5, 2), DATA3D(2, 5, 2), DATA3D(3, 5, 2), DATA3D(4, 5, 2), DATA3D(5, 5, 2));
                                 }
                                 ymtmp[2] =  POLINT_MACRO_QUINTIC_R2(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5]);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 3), DATA3D(1, 0, 3), DATA3D(2, 0, 3), DATA3D(3, 0, 3), DATA3D(4, 0, 3), DATA3D(5, 0, 3), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 3), DATA3D(1, 1, 3), DATA3D(2, 1, 3), DATA3D(3, 1, 3), DATA3D(4, 1, 3), DATA3D(5, 1, 3), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 3), DATA3D(1, 2, 3), DATA3D(2, 2, 3), DATA3D(3, 2, 3), DATA3D(4, 2, 3), DATA3D(5, 2, 3), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 3), DATA3D(1, 3, 3), DATA3D(2, 3, 3), DATA3D(3, 3, 3), DATA3D(4, 3, 3), DATA3D(5, 3, 3), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 3), DATA3D(1, 4, 3), DATA3D(2, 4, 3), DATA3D(3, 4, 3), DATA3D(4, 4, 3), DATA3D(5, 4, 3), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 3), DATA3D(1, 5, 3), DATA3D(2, 5, 3), DATA3D(3, 5, 3), DATA3D(4, 5, 3), DATA3D(5, 5, 3), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 3), DATA3D(1, 0, 3), DATA3D(2, 0, 3), DATA3D(3, 0, 3), DATA3D(4, 0, 3), DATA3D(5, 0, 3));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 3), DATA3D(1, 1, 3), DATA3D(2, 1, 3), DATA3D(3, 1, 3), DATA3D(4, 1, 3), DATA3D(5, 1, 3));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 3), DATA3D(1, 2, 3), DATA3D(2, 2, 3), DATA3D(3, 2, 3), DATA3D(4, 2, 3), DATA3D(5, 2, 3));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 3), DATA3D(1, 3, 3), DATA3D(2, 3, 3), DATA3D(3, 3, 3), DATA3D(4, 3, 3), DATA3D(5, 3, 3));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 3), DATA3D(1, 4, 3), DATA3D(2, 4, 3), DATA3D(3, 4, 3), DATA3D(4, 4, 3), DATA3D(5, 4, 3));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 3), DATA3D(1, 5, 3), DATA3D(2, 5, 3), DATA3D(3, 5, 3), DATA3D(4, 5, 3), DATA3D(5, 5, 3));
                                 }
                                 ymtmp[3] =  POLINT_MACRO_QUINTIC_R2(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5]);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 4), DATA3D(1, 0, 4), DATA3D(2, 0, 4), DATA3D(3, 0, 4), DATA3D(4, 0, 4), DATA3D(5, 0, 4), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 4), DATA3D(1, 1, 4), DATA3D(2, 1, 4), DATA3D(3, 1, 4), DATA3D(4, 1, 4), DATA3D(5, 1, 4), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 4), DATA3D(1, 2, 4), DATA3D(2, 2, 4), DATA3D(3, 2, 4), DATA3D(4, 2, 4), DATA3D(5, 2, 4), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 4), DATA3D(1, 3, 4), DATA3D(2, 3, 4), DATA3D(3, 3, 4), DATA3D(4, 3, 4), DATA3D(5, 3, 4), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 4), DATA3D(1, 4, 4), DATA3D(2, 4, 4), DATA3D(3, 4, 4), DATA3D(4, 4, 4), DATA3D(5, 4, 4), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 4), DATA3D(1, 5, 4), DATA3D(2, 5, 4), DATA3D(3, 5, 4), DATA3D(4, 5, 4), DATA3D(5, 5, 4), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 4), DATA3D(1, 0, 4), DATA3D(2, 0, 4), DATA3D(3, 0, 4), DATA3D(4, 0, 4), DATA3D(5, 0, 4));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 4), DATA3D(1, 1, 4), DATA3D(2, 1, 4), DATA3D(3, 1, 4), DATA3D(4, 1, 4), DATA3D(5, 1, 4));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 4), DATA3D(1, 2, 4), DATA3D(2, 2, 4), DATA3D(3, 2, 4), DATA3D(4, 2, 4), DATA3D(5, 2, 4));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 4), DATA3D(1, 3, 4), DATA3D(2, 3, 4), DATA3D(3, 3, 4), DATA3D(4, 3, 4), DATA3D(5, 3, 4));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 4), DATA3D(1, 4, 4), DATA3D(2, 4, 4), DATA3D(3, 4, 4), DATA3D(4, 4, 4), DATA3D(5, 4, 4));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 4), DATA3D(1, 5, 4), DATA3D(2, 5, 4), DATA3D(3, 5, 4), DATA3D(4, 5, 4), DATA3D(5, 5, 4));
                                 }
                                 ymtmp[4] =  POLINT_MACRO_QUINTIC_R2(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5]);
                                 if (Rx > 2) {
                                    yntmp[0] = POLINT_MACRO_QUINTIC(DATA3D(0, 0, 5), DATA3D(1, 0, 5), DATA3D(2, 0, 5), DATA3D(3, 0, 5), DATA3D(4, 0, 5), DATA3D(5, 0, 5), Rx, px);
                                    yntmp[1] = POLINT_MACRO_QUINTIC(DATA3D(0, 1, 5), DATA3D(1, 1, 5), DATA3D(2, 1, 5), DATA3D(3, 1, 5), DATA3D(4, 1, 5), DATA3D(5, 1, 5), Rx, px);
                                    yntmp[2] = POLINT_MACRO_QUINTIC(DATA3D(0, 2, 5), DATA3D(1, 2, 5), DATA3D(2, 2, 5), DATA3D(3, 2, 5), DATA3D(4, 2, 5), DATA3D(5, 2, 5), Rx, px);
                                    yntmp[3] = POLINT_MACRO_QUINTIC(DATA3D(0, 3, 5), DATA3D(1, 3, 5), DATA3D(2, 3, 5), DATA3D(3, 3, 5), DATA3D(4, 3, 5), DATA3D(5, 3, 5), Rx, px);
                                    yntmp[4] = POLINT_MACRO_QUINTIC(DATA3D(0, 4, 5), DATA3D(1, 4, 5), DATA3D(2, 4, 5), DATA3D(3, 4, 5), DATA3D(4, 4, 5), DATA3D(5, 4, 5), Rx, px);
                                    yntmp[5] = POLINT_MACRO_QUINTIC(DATA3D(0, 5, 5), DATA3D(1, 5, 5), DATA3D(2, 5, 5), DATA3D(3, 5, 5), DATA3D(4, 5, 5), DATA3D(5, 5, 5), Rx, px);
                                 }
                                 else {
                                    yntmp[0] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 0, 5), DATA3D(1, 0, 5), DATA3D(2, 0, 5), DATA3D(3, 0, 5), DATA3D(4, 0, 5), DATA3D(5, 0, 5));
                                    yntmp[1] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 1, 5), DATA3D(1, 1, 5), DATA3D(2, 1, 5), DATA3D(3, 1, 5), DATA3D(4, 1, 5), DATA3D(5, 1, 5));
                                    yntmp[2] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 2, 5), DATA3D(1, 2, 5), DATA3D(2, 2, 5), DATA3D(3, 2, 5), DATA3D(4, 2, 5), DATA3D(5, 2, 5));
                                    yntmp[3] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 3, 5), DATA3D(1, 3, 5), DATA3D(2, 3, 5), DATA3D(3, 3, 5), DATA3D(4, 3, 5), DATA3D(5, 3, 5));
                                    yntmp[4] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 4, 5), DATA3D(1, 4, 5), DATA3D(2, 4, 5), DATA3D(3, 4, 5), DATA3D(4, 4, 5), DATA3D(5, 4, 5));
                                    yntmp[5] = POLINT_MACRO_QUINTIC_R2(DATA3D(0, 5, 5), DATA3D(1, 5, 5), DATA3D(2, 5, 5), DATA3D(3, 5, 5), DATA3D(4, 5, 5), DATA3D(5, 5, 5));
                                 }
                                 ymtmp[5] =  POLINT_MACRO_QUINTIC_R2(yntmp[0], yntmp[1], yntmp[2], yntmp[3], yntmp[4], yntmp[5]);
                              }
                              if (Rz > 2) {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC(ymtmp[0], ymtmp[1], ymtmp[2], ymtmp[3], ymtmp[4], ymtmp[5], Rz, pz);
                              }
                              else {
                                 vector3D(fieldf, i - filo[0], j - filo[1], k - filo[2]) =  POLINT_MACRO_QUINTIC_R2(ymtmp[0], ymtmp[1], ymtmp[2], ymtmp[3], ymtmp[4], ymtmp[5]);
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      } else {
         TBOX_ERROR("LagrangianPolynomicRefine error...\n"
            << "dimension > 3 or < 2 not supported." << endl);

      }
   }
}
