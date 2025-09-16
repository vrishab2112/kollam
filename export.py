import json, csv
S=json.load(open('kolam29_37_spline.json'))
T=S['t']; ax, bx, cx, dx=[S['x'][k] for k in 'abcd']; ay, by, cy, dy=[S['y'][k] for k in 'abcd']
N=4000; i=0; pts=[]
for k in range(N):
    tt=T[0]+(T[-1]-T[0])*k/(N-1)
    while i < len(T)-2 and tt > T[i+1]: i+=1
    u=tt-T[i]
    x=ax[i]+bx[i]*u+cx[i]*u*u+dx[i]*u*u*u
    y=ay[i]+by[i]*u+cy[i]*u*u+dy[i]*u*u*u
    pts.append((x,y))
with open('kolam_points.csv','w',newline='') as f:
    csv.writer(f).writerows(pts)
print('wrote kolam_points.csv')