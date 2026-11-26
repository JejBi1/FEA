/*     CalculiX - A 3-dimensional finite element program                 */
/*              Copyright (C) 1998 Guido Dhondt                          */
/*     This program is free software; you can redistribute it and/or     */
/*     modify it under the terms of the GNU General Public License as    */
/*     published by the Free Software Foundation; either version 2 of    */
/*     the License, or (at your option) any later version.               */

/*     This program is distributed in the hope that it will be useful,   */
/*     but WITHOUT ANY WARRANTY; without even the implied warranty of    */
/*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the      */
/*     GNU General Public License for more details.                      */

/*     You should have received a copy of the GNU General Public License */
/*     along with this program; if not, write to the Free Software       */
/*     Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.         */

/* FEA Studio change (new file, 23. 11. 2026): interface between spooles.c and the faer solver (faer.c). */

#ifndef __CCX_FAER_H
#define __CCX_FAER_H

#include "CalculiX.h"

/* 1 = the system of equations is factorized by faer (and checked): faerc_solve / faerc_cleanup follow; 0 = not applicable
   or failed (the reason is in the log): SPOOLES factorizes it */
int faerc_try_factor(double *ad, double *au, double *adb, double *aub, double *sigma, ITG *icol, ITG *irow,
                     ITG *neq, ITG *nzs, ITG *symmetryflag, ITG *inputformat, ITG *nzs3);
int faerc_active(void);
void faerc_solve(double *b, ITG *neq);
void faerc_cleanup(void);
/* the line of `ccx -v` that the program reads (fea_core.ccx.runner) */
void faerc_marker(void);

#endif
