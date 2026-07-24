
#
# Copyright (C) 2021 Manuel Cotelo Ferreiro (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)
#
# Module with utility functions to manage module/program info and 
# run test inside the module
#
# Version control: 
# 2025.09.30: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - use dictionaries to store info and defs variables
# 2024.09.03: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - add defs variable
# - remove progress messages, not working fine
# 2024.07.05: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - redirects stderr to /dev/null 
# - fix bug in signed_pickle
# 2023.11.02: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - minor changes
# 2023.10.04: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - include functions to sign pickle files
# 2023.01.10: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - minor changes in disclaimer
# 2022.10.04: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - minor improvements
# 2022.07.27: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - change function names
# 2022.01.18: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - include GIT reversion info
# 2021.12.30: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - include date and time 
# 2021.10.21: Manuel Cotelo Ferreiro (manuel.cotelo@upm.es)
# - Initial version
#

import os
import sys
import argparse
import datetime
import shlex
import subprocess
import pickle
import hmac
import getpass
import hashlib
import secrets

# 
# script info
#

info = argparse.Namespace(
  name = 'utils',
  desc = 'Module with utility functions',
  author = 'Manuel Cotelo Ferreiro',
  email = 'manuel.cotelo@upm.es',
  year = 2024,
  version = [ 1, 0, 4, ],
  copyright = 'Copyright (C) 2021 Manuel Cotelo Ferreiro (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)',
)

defs = argparse.Namespace(
  verbose = 0,
)

# 
# functions to write info messages
#

def write_dict( d, ):
  return '\n'.join([ '# {:>18s} = {}'.format(ki,vi) for ki, vi in d.items() ])

def write_ns( ns, use_filter=False, ):
  if use_filter:
    return write_dict( { name: value for name, value in vars(ns).items() if not name.startswith('_') })
  else:
    return write_dict(vars(ns))

def write_data( data, ):
  if type(data) == dict:
    return write_dict( data, )
  elif type(data) == argparse.Namespace:
    return write_ns( data, )
  else:
    return str(data)

def write_info( info, ):
  return '\n# info:\n' + write_data(info)

def write_args( args, ):
  if args:
    return '\n# args:\n' + write_data(args)
  return ''

def write_date():
  return '\n# date:\n' + write_dict( { 'now': datetime.datetime.now(), }, )

def write_git():
  cmmd = 'git log -1'
  proc = subprocess.Popen(
    shlex.split(cmmd), 
    stdout=subprocess.PIPE, 
    stderr=subprocess.DEVNULL, 
  )
  outs, errs = proc.communicate()
  ls = outs.decode('utf8').splitlines()
  # get commit hash
  try:
    hash = ls[0].split()[1].strip()
    author = ls[1].split(' ',1)[1].strip()
    d = ls[2].split(' ',1)[1].strip()
    return '\n# GIT revision:\n' + write_dict( { 'hash': hash, 'author': author, 'date': d, }, )
  except:
    return '\n# GIT revision:\n' + write_dict( { '': 'not available', }, )

def write_init( info, args={}, ):
  return write_info(info) + '\n' + write_args(args) + '\n' + write_date() + '\n' + write_git() + '\n'

def write_disclaimer( info, ):
  if type(info) is dict:
    return f'# {info["copyright"]} hereby claims all interest in program "{info["name"]}"'
  elif type(info) is argparse.Namespace:
    return f'# {info.copyright} hereby claims all interest in program "{info.name}"'
  else:
    return '# Copyright (C) 2021 Manuel Cotelo Ferreiro (Instituto de Fusión Nuclear Guillermo Velarde - Universidad Politécnica de Madrid) hereby claims all interest in this program'

def write_test( ls, ):
  return '\n# available test = ' + ', '.join(ls.keys()) + '\n'

#
# sign pickle files
#

# write unsigned data to pickle format
def unsigned_pickle( file_name, data, ):
  with open( file_name, 'wb', ) as fd:
    pickle.dump( data, fd, protocol=pickle.HIGHEST_PROTOCOL, )
  return

# write signed data to pickle format
def signed_pickle( file_name, data, ):

  # pickle data
  s = pickle.dumps( data, protocol=pickle.HIGHEST_PROTOCOL, )
  # create hash with user defined key
  digest =  hmac.new( bytes( getpass.getpass('Key: '), 'utf8', ), s, hashlib.blake2b, ).hexdigest()
  
  # save header and pickled data to binary file
  with open( file_name, 'wb', ) as fd:
    fd.write( bytes( f'{digest} {len(s)}\n', 'utf8', ), )
    fd.write( s, )
  
  return digest

# read unsigned data from pickle format
def unsigned_unpickle( file_name, ):
  with open( file_name, 'rb', ) as fd:
    return pickle.load( fd, )

# read signed data from pickle format
def signed_unpickle( file_name, ):
  
  def header_split(header):
    digest, length = header.split()
    return digest, int(length)

  # read data from binary file
  with open( file_name, 'rb', ) as fd:
    
    # read file header that packages original digest
    header = fd.readline().decode('utf8')
    digest, length = header_split(header)
    
    data = fd.read(length)

  # rebuild hash from data with user key
  expected_digest = hmac.new( bytes( getpass.getpass('Key: '), 'utf8', ), data, hashlib.blake2b, ).hexdigest()

  # check rebuilded hash with in file hash
  if not secrets.compare_digest( digest, expected_digest, ):
    print( '# error: Invalid signature', file=sys.stderr, )
    sys.stderr.flush()
    sys.exit(1)

  return pickle.loads( data, )

def apply_pickle( file_name, data, is_signed=False, ):
  if is_signed:
    return signed_pickle( file_name, data, )
  else:
    return unsigned_pickle( file_name, data, )

def apply_unpickle( file_name, is_signed=False, ):
  if is_signed:
    return signed_unpickle( file_name, )
  else:
    return unsigned_unpickle( file_name, )

# 
# other functions
#

def nmsanitizer(name):
  return name.strip().lower()

#
# functions for testing
#
def filter_tests( vars, key_test, ):
  return { name: func for name, func in vars.items() if name.startswith(key_test) and callable(func) }

def run_test( info, vars, key_test='test_', iosync=True, ):

  # create list of test functions
  list_test = filter_tests(vars,key_test)
  
  # print some info
  print(write_info(info))

  # print list of test
  print(write_test(list_test))

  # run test
  for name, func in list_test.items():

    # print test name
    print( f'\n# run test "{name}" ...', file=sys.stdout, )

    # run test
    func()

    # sync I/O
    if iosync:
      sys.stdout.flush()

  print() 

  return 0

def show_message():
  print('\n# warning :: this file is not intended for standalone run, launch tests ...',file=sys.stderr,)
  return
 

#
# testing
#

def test_pickle_unsigned():
  
  file_name = 'test.pkl'

  data = [ 'a', 1, 2, 3, 'b', 'c', ]

  print( f'# pickle data and write to file', )
  print( f'# data: {data}', )
  unsigned_pickle( file_name, data, )
  print()
  
  print( f'# read data and unpickle unsigned data', )
  data_new = unsigned_unpickle( file_name, )
  print( f'# data_new: {data_new}', )
  print()

  os.remove(file_name)

  return

def test_pickle_signed():
  
  file_name = 'test.pkl'

  data = [ 'a', 1, 2, 3, 'b', 'c', ]

  print( f'# pickle data and write to file', )
  print( f'# data: {data}', )
  digest = signed_pickle( file_name, data, )
  print( f'# digest: {digest}', )
  print()
  
  print( f'# read data from signed pickle in file', )
  data_new = signed_unpickle( file_name, )
  print( f'# data_new: {data_new}', )
  print()

  os.remove(file_name)

  return

# run this script
if __name__ == '__main__':
  show_message()
  sys.exit( run_test( info, globals(), ), )
else:
  print( write_disclaimer(info), )


