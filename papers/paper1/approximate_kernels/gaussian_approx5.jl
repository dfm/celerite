# Try to approximate squared-exponential kernel:
using PyPlot
using Optim

function gaussian_approx5()

ntime = 1000
time = collect(linspace(0,10,ntime))
gaussian=exp(-time.^2/2.)

function compute_chi(x)
model = zeros(ntime)
c1 = 1.0
c2 = 0.0
for k=1:nexp
  model += x[1+(k-1)*2].*exp(-x[2+(k-1)*2].*time)
  c1 -= x[1+(k-1)*2]
  c2 -= x[1+(k-1)*2]*x[2+(k-1)*2]
end
c2 = c2/c1
model += c1.*exp(-c2.*time)
return sum((gaussian-model).^2)
end  

nexp = 6
#nexp = 2
#nexp = 4
npar = 2*nexp
abest=zeros(npar)
abest=[2.7518074963750734,1.5312499407621147,-0.3546256365636662,4.753504292338284,-1.3968344706386124,1.7967431129812566,1.0,0.5,0.1,0.1,-0.2,0.5]


atrial = zeros(npar)
for k=1:npar
  atrial[k] = abest[k]
end
model = zeros(ntime)
c1 = 1.0
c2 = 0.0
for k=1:nexp-1
  model += atrial[1+(k-1)*2].*exp(-atrial[2+(k-1)*2].*time)
  c1 -= atrial[1+(k-1)*2]
  c2 -= atrial[1+(k-1)*2]*atrial[2+(k-1)*2]
end
c2 = c2/c1
model += c1.*exp(-c2.*time)
clf()
plot(time,gaussian)
plot(time,model)
#chibest=sum((gaussian-model).^2)
chibest=compute_chi(atrial)

println("Initial chi-square: ",chibest)
read(STDIN,Char)

result = optimize(compute_chi, atrial, BFGS(), OptimizationOptions(autodiff = true))

println(Optim.minimizer(result),Optim.minimum(result))
abest= Optim.minimizer(result)
clf()
plot(time,gaussian)
plot(time,model)
model = zeros(ntime)
c1 = 1.0
c2 = 0.0
for k=1:nexp
  model += abest[1+(k-1)*2].*exp(-abest[2+(k-1)*2].*time)
  c1 -= abest[1+(k-1)*2]
  c2 -= abest[1+(k-1)*2]*abest[2+(k-1)*2]
end
c2 = c2/c1
model += c1.*exp(-c2.*time)
plot(time,(gaussian-model))
chi=sum((gaussian-model).^2)
println("New minimum: ",chi," ",abest," ",c1," ",c2," ",std(gaussian-model)," ",maximum(abs(gaussian-model)))
read(STDIN,Char)

return
end
