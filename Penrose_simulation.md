
# Ray Simulation in Penrose cavity - Juman @ QFLL (2018.12.22)
## Just Drag and drop the points in the PSOS and look what happens in the ray dynamics

Jupyter notebook은 공동으로 python 작업을 하는경우에 적합한 컴파일러로 볼 수 있습니다. 개발자들은 이 노트북을 GitHub이나 Anaconda cloud에 올려 다른 사람들과 공유하고 웹페이지에서 코드를 실행시킬 수 있게 만들곤 합니다. 후자의 경우 제가 방법을 모르기 때문에 interactive plot의 실행을 위해 파이썬 개발환경을 만드는 것부터 시작하겠습니다.
이번 주에는 Jupyter notebook 에서 bloomberg의 bqplot을 활용하여 PSOS 상에서 point를 drag and drop하면서 ray simulation을 할 수 있도록 만드는 작업을 하였습니다. 우선 bqplot 과 ipywidgets 을 import해야하기 때문에 설치를 합니다.



## 1. Python 개발환경 만들기
1. https://www.anaconda.com/download/ 에 접속하여 anaconda의 가장 최신 버전(현재 ver 3.7)을 설치합니다. 모두 기본설정으로 하고 Next를 계속 누릅니다.
2. 설치가 끝나면 Anaconda Prompt 를 실행시키고 conda install -c conda-forge bqplot 를 쓰고 엔터를 누릅니다.![image.png](attachment:image.png)

3. bqplot 설치가 끝나면 anaconda navigator를 실행시킵니다.
4. 처음 페이지를 보시면 jupyter notebook이 있는데 Launch 를 누릅니다.![image.png](attachment:image.png)
5. 노트북이 나오면 첫줄에 다음과 같이 적고 ctrl+Enter를 누릅니다.(cell을 실행한다는 뜻입니다. shift+Enter를 칠 경우 cell을 실행하고 다음 cell로 이동합니다.)




```python
import sys
sys.executable
```




    'C:\\Users\\kjman\\AppData\\Local\\Continuum\\anaconda3\\python.exe'



6. 결과로 나온 경로는 파이썬이 실행될 때 module을 가져오는 폴더입니다. anaconda3의 하위 폴더 중 Lib\site-packages 폴더로 갑니다.
7. site-packages 폴더 안에 제가 드린 dynamical_barrier.py 를 붙여넣습니다. 
8. Jupyter notebook 용 폴더를 하나 만들어서 그 안에 Penrose_simulation.ipynb 파일을 넣고 엽니다.
9. Kernel에서 Restart&Run All 을 누르시면 interactive plot을 실행할 수 있습니다.

## 2. bqplot 사용하기



```python
from __future__ import print_function
import numpy as np
from bqplot import Axis,LinearScale,Scatter,Lines,Figure
from bqplot import *
from ipywidgets import Label,HBox,VBox,FloatSlider,IntSlider,Play,jslink
import dynamical_barrier as db
import math
import os
from tqdm import tqdm
if not os.path.exists('Penrose_Temp'):
    os.mkdir('Penrose_Temp')
#Thanks to Bloomberg for bqplot module and guide
```

### Penrose cavity의 geometry에 해당하는 전역변수입니다.


```python
a=3.
b=4.8
a1=3.
b1=1.945
c=0.9
k = np.sqrt(np.absolute(a**2-b**2))
```


```python
x_data=[0.5]
y_data=[0.7]
sc_x = LinearScale(min=0.0,max=1.0)
sc_y = LinearScale(min=-1.0,max=1.0)
sc_x2=LinearScale(min=-6,max=6)
sc_y2=LinearScale(min=-4.8,max=4.8)

#PSOS
scat = Scatter(x=x_data, y=y_data, scales={'x': sc_x, 'y': sc_y}, colors=['orange'],enable_move=True)
#Penrose
scat2 = Scatter(x=[], y=[], scales={'x': sc_x2, 'y': sc_y2},colors=['orange'])
```

### focdist는 초점 근처를 자른 길이, h1,h2는 PSOS 상에 정사각형 영역을 잡을 때 수평방향 길이와 수직방향 길이, q는 영역 안에 있는 initial condition들의 개수입니다.


```python
focdist=0.1
h1=0.001
h2=0.001
q=5
new_length=b1*np.sqrt(1-((a1-focdist)/a1)**2)
```

### Penrose 그림그리기


```python

def plot_ellipse(a,b,xdis,ydis,iang,fang):
    q=100
    step=(fang-iang)/q
    x=np.ones(q+1)
    y=np.ones(q+1)
    
    for t in range(q+1):
        theta=iang+step*t
        x[t]*=(xdis+a*np.cos(theta))
        y[t]*=(ydis+b*np.sin(theta))
    return x,y
wall_lines=[]
#Wall
wall_lines.append(Lines(x=[-a1,-c],y=[b,b],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['red'],stroke_width=5))
wall_lines.append(Lines(x=[-a1+focdist,-c],y=[k,k],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['red'],stroke_width=5))
wall_lines.append(Lines(x=[-a1+focdist,-a1+focdist],y=[k,k-new_length],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['black'],stroke_width=5))
wall_lines.append(Lines(x=[a1-focdist,a1-focdist],y=[k,k-new_length],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['black'],stroke_width=5))
wall_lines.append(Lines(x=[-a1+focdist,-a1+focdist],y=[-k,-k+new_length],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['black'],stroke_width=5))
wall_lines.append(Lines(x=[a1-focdist,a1-focdist],y=[-k,-k+new_length],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['black'],stroke_width=5))
wall_lines.append(Lines(x=[c,a1],y=[b,b],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['blue'],stroke_width=5))
wall_lines.append(Lines(x=[c,a1-focdist],y=[k,k],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['blue'],stroke_width=5))
wall_lines.append(Lines(x=[-a1+focdist,-c],y=[-k,-k],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['red'],stroke_width=5))
wall_lines.append(Lines(x=[-a1,-c],y=[-b,-b],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['red'],stroke_width=5))
wall_lines.append(Lines(x=[c,a1-focdist],y=[-k,-k],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['blue'],stroke_width=5))
wall_lines.append(Lines(x=[c,a1],y=[-b,-b],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['blue'],stroke_width=5))
wall_lines.append(Lines(x=[-c,-c],y=[k,b],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['red'],stroke_width=5))
wall_lines.append(Lines(x=[-c,-c],y=[-k,-b],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['red'],stroke_width=5))
wall_lines.append(Lines(x=[c,c],y=[k,b],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['blue'],stroke_width=5))
wall_lines.append(Lines(x=[c,c],y=[-k,-b],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['blue'],stroke_width=5))
wall_lines.append(Lines(x=plot_ellipse(a,b,-a1,0,np.pi/2,3*np.pi/2)[0],y=plot_ellipse(a,b,-a1,0,np.pi/2,3*np.pi/2)[1],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['green'],stroke_width=5))
wall_lines.append(Lines(x=plot_ellipse(a,b,a1,0,-np.pi/2,np.pi/2)[0],y=plot_ellipse(a,b,a1,0,-np.pi/2,np.pi/2)[1],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['green'],stroke_width=5))
wall_lines.append(Lines(x=plot_ellipse(a1,b1,0,-k,math.acos((a1-focdist)/a1),np.pi-math.acos((a1-focdist)/a1))[0],y=plot_ellipse(a1,b1,0,-k,math.acos((a1-focdist)/a1),np.pi-math.acos((a1-focdist)/a1))[1],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['green'],stroke_width=5))
wall_lines.append(Lines(x=plot_ellipse(a1,b1,0,k,np.pi+math.acos((a1-focdist)/a1),2*np.pi-math.acos((a1-focdist)/a1))[0],y=plot_ellipse(a1,b1,0,k,np.pi+math.acos((a1-focdist)/a1),2*np.pi-math.acos((a1-focdist)/a1))[1],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['green'],stroke_width=5))

```

### 반복해서 reflection 시키고 local 경로에 Point_Temp 폴더를 만들어서 point  save and load


```python
rays=[]
PSOS_points=[]
for i in range(q*q):
    rays.append(Lines(x=[],y=[],scales={'x':sc_x2,'y':sc_y2},line_style='solid',colors=['magenta'],stroke_width=1,animation_duration=100))
    PSOS_points.append(Scatter(x=[], y=[], scales={'x': sc_x, 'y': sc_y}, colors=['orange'],default_size=10))
#irayset={'n':np.array([np.zeros(4) for i in range(q*q)])}
def update_point(change=None):
    with scat2.hold_sync():
        irayset=np.array(db.IC_generator(scat.x[0],scat.y[0],focdist,h1,h2,q,q))
        scat2.x=irayset.T[0]
        scat2.y=irayset.T[1]
        for i in range(q*q):
            fray=db.reflected_ray(irayset[i],focdist)
            rays[i].x=[irayset[i][0],fray[0]]
            rays[i].y=[irayset[i][1],fray[1]]


def end_func(self,content):
    irayset=np.array(db.IC_generator(scat.x[0],scat.y[0],focdist,h1,h2,q,q))
    for t in tqdm(range(101),desc='Now loading'):
        fp=open(os.getcwd()+'\\Penrose_Temp\\'+str(t)+'.txt','w')
        fp1=open(os.getcwd()+'\\Penrose_Temp\\'+'PSOS'+str(t)+'.txt','w')
        for i in range(q*q):
            fray=db.reflected_ray(irayset[i],focdist)
            ang_diff=db.nangle_pt(fray[0],fray[1],fray[3],focdist)-fray[2]
            DB=db.Dynamical_Barrier(fray[0],fray[1],ang_diff,fray[3],focdist)
            data=[db.eta(fray[0],fray[1],fray[3],focdist)/db.eta(0,0,14,focdist),np.sin(ang_diff)]
            fp.write('%f %f %f %f\n' % (fray[0],fray[1],fray[2],fray[3]))
            fp1.write('%f %f %d\n' % (data[0],data[1],DB))
            for m in range(4):
                irayset[i][m]=fray[m]
        fp.close()
        fp1.close()
scat.on_drag_end(end_func)
#def update_focdist(change=None):
#scat.update_on_move=True    
update_point()

```


```python
step_slider=IntSlider(min=0,max=100,step=1,description='Step',value=0)
def ray_simulation(change):
    with step_slider.hold_sync():
        k=step_slider.value
        iray=open(os.getcwd()+'\\Penrose_Temp\\'+str(k)+'.txt').read().split()
        fray=open(os.getcwd()+'\\Penrose_Temp\\'+str(k+1)+'.txt').read().split()
        dbiray=open(os.getcwd()+'\\Penrose_Temp\\'+'PSOS'+str(k)+'.txt').read().split()
        dbfray=open(os.getcwd()+'\\Penrose_Temp\\'+'PSOS'+str(k+1)+'.txt').read().split()
        iray=list(map(float,iray))
        fray=list(map(float,fray))
        dbiray=list(map(float,dbiray))
        dbfray=list(map(float,dbfray))
        for i in range(q*q):
                rays[i].x=[iray[4*i+0],fray[4*i+0]]
                rays[i].y=[iray[4*i+1],fray[4*i+1]]
                PSOS_points[i].x=[dbfray[3*i+0]]
                PSOS_points[i].y=[dbfray[3*i+1]]
                if dbfray[3*i+2]==1:
                    PSOS_points[i].colors=['red']
                if dbfray[3*i+2]==2:
                    PSOS_points[i].colors=['blue']
                if dbfray[3*i+2]==3:
                    PSOS_points[i].colors=['green']
                if dbfray[3*i+2]==4:
                    PSOS_points[i].colors=['black']
                    
                
step_slider.observe(ray_simulation,'value')            
play_button = Play(min=0, max=100,interval=100)
jslink((play_button,'value'),(step_slider,'value'))

# update line on change of x or y of scatter
scat.observe(update_point, names=['x'])
scat.observe(update_point, names=['y'])
ax_x = Axis(scale=sc_x, tick_format='0.1f', orientation='horizontal')
ax_y = Axis(scale=sc_y, tick_format='0.1f', orientation='vertical')
ax_x2=Axis(scale=sc_x2, orientation='horizontal',visible=False)
ax_y2=Axis(scale=sc_y2, orientation='vertical',visible=False)
fig = Figure(marks=[scat]+PSOS_points, axes=[ax_x, ax_y],fig_margin={'top':100,'bottom':30,'left':30,'right':50},title='PSOS')
fig2=Figure(marks=[scat2]+wall_lines+rays,axes=[ax_x2,ax_y2],fig_margin={'top':100,'bottom':30,'left':30,'right':50},title='Ray_simulation')

VBox([HBox([play_button,step_slider]),HBox([fig2,fig])])
```


    VBox(children=(HBox(children=(Play(value=0), IntSlider(value=0, description='Step'))), HBox(children=(Figure(a…


### 1. PSOS 상에서 orange point를 drag and drop해 원하는 곳에 놓으면 Now loading 이 뜹니다. 완료되기 전에 점을 다시 움직이면 오류가 생길 수 있습니다.
### 2. loading이 완료되면 재생버튼을 눌러 ray simulation을 돌릴 수 있습니다.
