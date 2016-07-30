using PyPlot
# Generate some data:

fig,ax = subplots()
ngen = 10
sig_best = zeros(ngen)

for igen=1:ngen
nt = 10

ttv = randn(nt)

# Now infer the uncertainty:

nsig = 100
sig = logspace(-0.5,1,nsig)


like_exp = zeros(nsig)
like_det = -0.5.*nt.*log(2pi.*sig.^2)
sig_best[igen] = 0.
like_best = -Inf
for i=1:nsig
  like_exp[i] = -0.5*sum((ttv./sig[i]).^2)
  if (like_exp[i]+like_det[i]) > like_best
    sig_best[igen]= sig[i]
    like_best = like_exp[i]+like_det[i]
  end
end


miny = minimum([like_det;like_exp;like_exp+like_det])
maxy = maximum([like_det;like_exp;like_exp+like_det])

ax[:semilogx](sig,like_exp,"r-",label = "exp",alpha=0.6)
ax[:semilogx](sig,like_det,"b-",label = "det",alpha=0.6)
ax[:semilogx](sig,like_det+like_exp,"k-",label = "det",alpha=0.6)
ax[:semilogx]([1,1],[miny,maxy],"m-",label="actual",linewidth=2,alpha=0.6)
ax[:semilogx]([sig_best[igen],sig_best[igen]],[miny,maxy],"c-",label="inferred",linewidth=2,alpha=0.6)
#ax[:legend](loc="lower right")
end
read(STDIN,Char)

plot(sig_best)
