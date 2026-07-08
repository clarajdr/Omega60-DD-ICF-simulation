
#
# Copyright (C) 2022 Manuel Cotelo Ferreiro (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)
# Author: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
#
# Module to define indexes for properties for an HDF5 file with 
# ARWEN output
#
# Version info:
# 2022.11.21: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - minor updates
#
 
import sys
import argparse

import numpy

import utils

# module info
info = argparse.Namespace(
  name = 'index',
  desc = 'Define indexes for ARWEN propoerties',
  author = 'Manuel Cotelo Ferreiro',
  email = 'manuel.cotelo@upm.es',
  year = 2022,
  version = [ 1, 1, 0, ],
  copyright = 'Copyright (C) 2022 Manuel Cotelo Ferreiro (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)',
)

# module defaults
defs = argparse.Namespace(
)

# create namespace with indices for state
def state_vars(num_mat):

  js = argparse.Namespace(
    
    num_mat = num_mat,
    
    # for XY
    xl = 0, xh = 1,
    yl = 2, yh = 3,
  
    # for RZ
    rl = 0, rh = 1,
    zl = 2, zh = 3,
  
    vol = 4,
  
    den = 5,
    
    # for XY
    ux = 6, uy = 7,
  
    # for RZ
    ur = 6, uz = 7,
    
    tem = 8,
    sde = 9,
  
    pre_thm = 10,
    pre_rad = 11,
    pre = 12,
    
    ene = 13,
    cv = 14, 
    rhocv = 15,
    cs = 16,
    zff = 17,
    eden = 18,
    ekin = 19,
    nm = 20,
    fvcolor = 21,
  )
 
  def _helper_svars(kref,num_mat,i):
    k = kref + i
    return argparse.Namespace(
      f   = k,
      fr  = k + 1*num_mat,
      fre = k + i + 2*num_mat,
      )

  kref = 22
  js.ms = [ _helper_svars(kref,num_mat,i) for i in range(num_mat) ]

  return js

# create namespace wiwth indioces for radiation
def radiation_vars(num_groups):

  kref = 6
  kbegin = kref
  kend = kbegin + num_groups

  jr = argparse.Namespace(
    
    num_groups = num_groups,
    
    # for XY
    xl = 0, xh = 1,
    yl = 2, yh = 3,
  
    # for RZ
    rl = 0, rh = 1,
    zl = 2, zh = 3,
  
    vol = 4,

    tem = 5,
    
    gs_begin = kbegin,
    gs_end = kend,
    
    gs = numpy.arange(kbegin,kend,dtype=int),

    gs_slice = numpy.s_[...,kbegin:kend],
    
    gtot = kend, 

    src = kend + 1,
    trad = kend + 2,
    rhocv = kend + 3,
    delta_tem = kend + 4,
  )
  
  return jr


# try to guess number of materials in HDF5 dataset for state variables
def guess_num_mat(ds):
  nm = 0
  for vi in ds.attrs.values():
    try:
      if 'fvrho' in vi.decode('utf8'):
        nm += 1
    except:
      pass
  return nm // 2

# try to guess number of radiation groups in a HDF5 dataset from 
# radiation variables
def guess_num_groups(ds):
  c = 0
  for vi in ds.attrs.values():
    try:
      if 'Rad.' in vi.decode('utf8'):
        c += 1
    except:
      pass
  return c

# 
# test
#

def test_index():
  for ki, vi in vars(state_vars(2)).items():
    print('# {:>18s} = {}'.format(ki,vi))
  return

# run this script
if __name__ == '__main__':
  utils.show_message()
  sys.exit( utils.run_test( info, globals(), ), )
else:
  print( utils.write_disclaimer(info), )