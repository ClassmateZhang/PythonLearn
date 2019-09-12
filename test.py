import redis
import time
import math
import sys

v=''
#for j in range(0,1024):
try:
    v+=sys.argv[1]
except IndexError:
    print('Please take parameters for text execution')
    sys.exit()
if len(v)==0:
    print('Please take parameters for text execution')
    sys.exit()
print (v)
r = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)
t1 = int(time.time())
#for i in range(0,100000):
mkey = v
m=r.hgetall(mkey)
t2 = int(time.time())
print (t2-t1)
#print m
r2 = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)
for i in m:
    #print i
    #print m[i]
    r2.hset(mkey,i,m[i])
print('succeed')