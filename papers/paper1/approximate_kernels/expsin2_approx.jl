# Try to approximate exponential-sine-squared kernel:
using PyPlot
using Optim

function expsin2_approx()
nexp = 5
ngamma = 100
coeff_grid = zeros(ngamma,nexp)
#gamma_grid = logspace(-2,0.5,ngamma)
gamma_grid = linspace(0.01,3.0,ngamma)
ntime = 200
time = collect(linspace(0,2pi,ntime))

gamma = 1.0
gaussian=exp(-gamma.*sin(time).^2)

function compute_chi(x)
model = zeros(ntime)
for k=1:nexp
  model += x[k].*cos((k-1)*2.*time)
end
return sum((gaussian-model).^2)
end  

npar = nexp

for i=1:ngamma
gamma = gamma_grid[i]
gaussian=exp(-gamma.*sin(time).^2)

abest=zeros(npar)
abest = [0.6,0.2,0.01,0.05,.025,.025]

atrial = zeros(npar)
for k=1:npar
  atrial[k] = abest[k]
end
model = zeros(ntime)
for k=1:nexp
  model += atrial[k].*cos((k-1)*2.*time)
end
clf()
plot(time,gaussian)
plot(time,model)
#chibest=sum((gaussian-model).^2)
chibest=compute_chi(atrial)

println("Initial chi-square: ",chibest)
#read(STDIN,Char)

result = optimize(compute_chi, atrial, BFGS(), OptimizationOptions(autodiff = true))

println(Optim.minimizer(result),Optim.minimum(result))
abest= Optim.minimizer(result)
clf()
plot(time,gaussian)
model = zeros(ntime)
for k=1:nexp
  model += abest[k].*cos((k-1)*2.*time)
end
plot(time,model)
plot(time,(gaussian-model))
chi=sum((gaussian-model).^2)
println("New minimum: ",chi," ",abest," ",std(gaussian-model)," ",maximum(abs(gaussian-model)))
#read(STDIN,Char)
coeff_grid[i,:] = abest
end

#coeff1 = log10(vec(coeff_grid[:,1]))

#function coeff1_fit(x)
## Fit the first coefficient:
#model= x[1]+x[2].*gamma_grid.^x[3]
#return sum((coeff1-model).^2)
#end

#x1 = [0.0,-1.75,0.2]
#result = optimize(coeff1_fit, x1, BFGS(), OptimizationOptions(autodiff = true))
#clf()
#loglog(gamma_grid,coeff_grid[:,1])
#x1 = Optim.minimizer(result)
##loglog(gamma_grid,10.^(-x1[2].*exp(log10(gamma_grid)*x1[1])),"r-")
#loglog(gamma_grid,10.^(x1[1]+x1[2].*gamma_grid.^x1[3]),"r-")
#println(Optim.minimizer(result),Optim.minimum(result))
#println("Fit to first coefficient")
#read(STDIN,Char)

# Now fit the grid of coefficients:
coeff2 = vec(coeff_grid[:,1])

ord = 1.0
function coeff2_fit(x)
# Fit the second-fourth coefficient:
model= x[1]+x[2].*exp(-gamma_grid.^x[3].*x[4])
return sum((coeff2-model).^2)
end

x1 = [0.45,0.4,1.0,1.0]
result = optimize(coeff2_fit, x1, BFGS(), OptimizationOptions(autodiff = true))
x1 = Optim.minimizer(result)
clf()
plot(gamma_grid,coeff2)
plot(gamma_grid,x1[1]+x1[2].*exp(-gamma_grid.^x1[3].*x1[4]))
println(Optim.minimizer(result),Optim.minimum(result))
println("Fit to first coefficient")
read(STDIN,Char)

coeff2 = vec(coeff_grid[:,2])
ord = 1.0
x2 = [0.45,0.4,1.0,1.0]
result = optimize(coeff2_fit, x2, BFGS(), OptimizationOptions(autodiff = true))
x2 = Optim.minimizer(result)
clf()
plot(gamma_grid,coeff2)
plot(gamma_grid,x2[1]+x2[2].*exp(-gamma_grid.^x2[3].*x2[4]))
println(Optim.minimizer(result),Optim.minimum(result))
println("Fit to second coefficient")
read(STDIN,Char)

coeff2 = vec(coeff_grid[:,3])
ord = 2.0
x3 = [0.45,0.4,1.0,1.0]
result = optimize(coeff2_fit, x3, BFGS(), OptimizationOptions(autodiff = true))
x3 = Optim.minimizer(result)
clf()
plot(gamma_grid,coeff2)
plot(gamma_grid,x3[1]+x3[2].*exp(-gamma_grid.^x3[3].*x3[4]))
println(Optim.minimizer(result),Optim.minimum(result))
println("Fit to third coefficient")
read(STDIN,Char)

coeff2 = vec(coeff_grid[:,4])
ord = 3.0
x4 = [0.45,0.4,1.0,1.0]
result = optimize(coeff2_fit, x4, BFGS(), OptimizationOptions(autodiff = true))
x4 = Optim.minimizer(result)
clf()
plot(gamma_grid,coeff2)
plot(gamma_grid,x4[1]+x4[2].*exp(-gamma_grid.^x4[3].*x4[4]))
println(Optim.minimizer(result),Optim.minimum(result))
println("Fit to fourth coefficient")
read(STDIN,Char)

coeff2 = vec(coeff_grid[:,5])
ord = 4.0
x5 = [0.45,0.4,1.0,1.0]
result = optimize(coeff2_fit, x5, BFGS(), OptimizationOptions(autodiff = true))
x5 = Optim.minimizer(result)
clf()
plot(gamma_grid,coeff2)
plot(gamma_grid,x5[1]+x5[2].*exp(-gamma_grid.^x5[3].*x5[4]))
println(Optim.minimizer(result),Optim.minimum(result))
println("Fit to fifth coefficient")
read(STDIN,Char)

# Now test with random numbers:
ntest = 10000
dev = zeros(ntest)
for i=1:ntest
  time = rand()*2pi
  gamma = rand()*3.0
  model = 0.0
  model += x1[1]+x1[2].*exp(-gamma.^x1[3].*x1[4])
  model += (x2[1]+x2[2].*exp(-gamma.^x2[3].*x2[4]))*cos(2*time)
  model += (x3[1]+x3[2].*exp(-gamma.^x3[3].*x3[4]))*cos(4*time)
  model += (x4[1]+x4[2].*exp(-gamma.^x4[3].*x4[4]))*cos(6*time)
  model += (x5[1]+x5[2].*exp(-gamma.^x5[3].*x5[4]))*cos(8*time)
  value = exp(-gamma*sin(time)^2)
  dev[i]=value-model
  println(i," ",gamma," ",time," ",value," ",model," ",value-model)
end
println("Maximum deviation:  ",maximum(abs(dev)))
println("Standard deviation: ",std(dev))
read(STDIN,Char)

return gamma_grid,x1,x2,x3,x4,x5
end
