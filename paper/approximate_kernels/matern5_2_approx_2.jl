# Try to approximate squared-exponential kernel:
using PyPlot
using Optim

function matern5_2_approx_2()

ntime = 1000
time = collect(linspace(0,10,ntime))
matern52=(1.0+time+time.^2./3.).*exp(-time)
nexp = 3

function compute_chi(x)
model = zeros(ntime)
for k=1:nexp
  model += x[1+(k-1)*2].*exp(-x[2+(k-1)*2].*time)
end
return sum((matern52-model).^2)
end  


# If we want to approximate Gaussian, then we just get a cosine... that doesn't work well!

# How about sum of two?  This I solved numerically with Mathematically, and unfortunately I find
# a growing solution in each case... darn.

npar = 2*nexp
abest=zeros(npar)
abest[1:6]=[-9.,1.,10.,0.9,0.1,0.1]
atrial = zeros(npar)
for k=1:npar
  atrial[k] = abest[k]
end
model = zeros(ntime)
for k=1:nexp
  model += atrial[1+(k-1)*2].*exp(-atrial[2+(k-1)*2].*time)
end
clf()
plot(time,matern52)
plot(time,model)
#chibest=sum((matern52-model).^2)
chibest=compute_chi(atrial)

println("Initial chi-square: ",chibest)
read(STDIN,Char)

result = optimize(compute_chi, atrial, BFGS(), OptimizationOptions(autodiff = true))

println(Optim.minimizer(result),Optim.minimum(result))
atrial = Optim.minimizer(result)
clf()
plot(time,(1+time+time.^2/3).*exp(-time))
plot(time,model)
model = zeros(ntime)
for k=1:nexp
  model += atrial[1+(k-1)*2].*exp(-atrial[2+(k-1)*2].*time)
end
chi=sum((matern52-model).^2)
plot(time,(matern52-model))
println("New minimum: ",chi," ",atrial," ",std(matern52-model))
read(STDIN,Char)

atrial = zeros(npar)
for i=1:1000
  for j=1:npar-2
    atrial[j] = abest[j] + 0.01*randn()
    atrial[7]= 1.0-atrial[1]-atrial[3]-atrial[5]
    atrial[8]=-(atrial[1]*atrial[2]+atrial[3]*atrial[4]+atrial[5]*atrial[6])/atrial[7]
#    atrial[5]= 1.0-atrial[1]-atrial[3]
#    atrial[6]=-(atrial[1]*atrial[2]+atrial[3]*atrial[4])/atrial[5]
    model = zeros(ntime)
    for k=1:nexp
      model += atrial[1+(k-1)*2].*exp(-atrial[2+(k-1)*2].*time)
    end
    chi=sum(((1.0+time+time.^2./3.0).*exp(-time)-model).^2)
    if chi < chibest
      chibest = chi
      for k=1:npar
        abest[k] = atrial[k]
      end
      println("New minimum: ",i," ",chibest," ",abest)
      clf()
      plot(time,(1+time+time.^2/3).*exp(-time))
      plot(time,model)
      plot(time,(1+time+time.^2/3).*exp(-time)-model)
    end
  end
end

end
