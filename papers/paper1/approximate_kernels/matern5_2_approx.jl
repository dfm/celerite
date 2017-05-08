# Try to approximate squared-exponential kernel:
using PyPlot
using Optim

function matern5_2_approx()

ntime = 1000
time = collect(linspace(0,10,ntime))
matern52=(1.0+time+time.^2./3.).*exp(-time)
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

#nexp = 3
nexp = 4
npar = 2*nexp
abest=zeros(npar)
#abest[1:6]=[-9.,1.,10.,0.9,0.1,1.1]
abest[1:6]=[-3.,1.,4.,0.75,0.1,1.1]
# Here are best parameters so far:
#New minimum: 7063 0.0009743003566026671 
#abest=[-4.081065227181365,1.0304513099118429,0.7384172905709343,1.6404128688104038,-0.6973216477808734,0.47273293933522775,5.0399695843913035,0.659463641544771]
#New minimum: 4091 0.0007528492057342187 
#abest=[-3.983277629878132,1.0735443595362018,0.9310909822220776,1.609723984179903,-0.728531695566164,0.4833054951112925,4.780718343222218,0.6546148284917492]
#New minimum: 3509 0.0006919748190206947 [-4.012577986446492,1.0702556640310126,0.9246366188173811,1.6094033319207477,-0.756585292781486,0.48539249997921724,4.844526660410597,0.655092253875969]
#abest=[-4.012577986446492,1.0702556640310126,0.9246366188173811,1.6094033319207477,-0.756585292781486,0.48539249997921724,4.844526660410597,0.655092253875969]
#New minimum: 99883 0.0004297805010976177 [-4.707752923722873,1.0471511183944253,1.0707583835332415,1.5530678690438415,-1.023610702073631,0.522600717254369,5.660605242263262,0.6716080388250383]
#abest=[-4.707752923722873,1.0471511183944253,1.0707583835332415,1.5530678690438415,-1.023610702073631,0.522600717254369,5.660605242263262,0.6716080388250383]
#New minimum: 99770 0.00034643674752663026 [-5.071392864979707,1.041877623915747,1.1886472038873641,1.521389181893939,-1.1341375315197493,0.5358293183936108,6.016883192612092,0.6786038154302075]
abest=[-5.071392864979707,1.041877623915747,1.1886472038873641,1.521389181893939,-1.1341375315197493,0.5358293183936108,6.016883192612092,0.6786038154302075]
#New minimum: 999639 0.00016786630230513215 [-6.52117872187123,1.0277101206108639,1.6861024710175314,1.4296430478894617,-1.5767703086820388,0.5777664250974277,7.4118465595357375,0.7018981839031324]
#abest=[-6.52117872187123,1.0277101206108639,1.6861024710175314,1.4296430478894617,-1.5767703086820388,0.5777664250974277,7.4118465595357375,0.7018981839031324]

#New minimum: 950075 0.0001478308089395661 [-6.80168281621302,1.0282509070629977,1.8226885846422511,1.4133527852506547,-1.7003897369362933,0.5856387342941094,7.679383968507063,0.7049456900610803]

#abest[1:4]=[-9.,1.,10.1,0.9]
abest[7]=1.0-abest[1]-abest[3]-abest[5]
abest[8]=-(abest[1]*abest[2]+abest[3]*abest[4]+abest[5]*abest[6])/abest[7]
#abest[5]=1.0-abest[1]-abest[3]
#abest[6]=-(abest[1]*abest[2]+abest[3]*abest[4])/abest[5]
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
#atrial = Optim.minimizer(result)
atrial = [-147.6965573,1.0097610,85.8875034,1.0621734,-4.6540017,0.8353745,67.4630744,0.9160394]
plot(time,matern52)
plot(time,model)
model = zeros(ntime)
for k=1:nexp
  model += atrial[1+(k-1)*2].*exp(-atrial[2+(k-1)*2].*time)
end
plot(time,(matern52-model))
chi=sum((matern52-model).^2)
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
