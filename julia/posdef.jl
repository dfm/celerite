# Make plot of positive definiteness of Gaussian approximation:
using PyPlot
using ForwardDiff

#function posdef()
ntime = 10000
time = collect(linspace(1e-2,1000,ntime))
#nc = 3
#amp = [2.7518074963750734,-0.3546256365636662,-1.3968344706386124]
#b = [1.5312499407621147,4.753504292338284,1.7967431129812566]
#omega = [0.34148888178746056,2.042571639590954,1.7641684073510928]
nc = 4
#amp = [2.953823582534888,-0.04961614737314587,-1.6900719772140012,-0.21413545794774125]
#b = [1.589510614489265,3.4729345651560872,1.8698492189074047,6.363451205766626]
#omega = [0.3329119766576696,4.5999414581901386,1.7115890219170917,2.205939216106191]

amp = [2.953823582534889, -0.04961614737314587, -1.6900719772140012, -0.21413545794774125]
b =[1.589510614489265, 3.4729345651560872, 1.8698492189074047, 6.363451205766626]
omega =[0.3329119766576696, 4.5999414581901386,1.7115890219170917, 2.205939216106191]

function func(x)
fout=0.0
for k=1:nc
  fout +=amp[k]*exp(-b[k]*sqrt(x))*cos(omega[k]*sqrt(x))
end
#fout=exp(-x*0.5)
return fout
end

y = zeros(ntime)
dy = zeros(ntime)
d2y = zeros(ntime)
for i=1:ntime
  y[i]=func(time[i])
  dy[i]=ForwardDiff.derivative(func,time[i])
  d2y[i]=ForwardDiff.derivative(z -> ForwardDiff.derivative(func,z),time[i])
end
plot(time,y)
plot(time,dy)
plot(time,d2y)

#return
#end
