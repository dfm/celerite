function compute_likelihood_save(time_phot,param,data,aex,al_small,indx,logdeta,p)

# First call prior:

#log_prior = compute_prior(param)
log_prior = 0.0

# Then call model (for now set model to zero):
model = zeros(length(time_phot))

# Then compute log-likelihood:

# Subtract off the model from the data:
y = data - model

log_like = lorentz_likelihood_hermitian_band_save(p,y,aex,al_small,indx,logdeta)

return log_like
end
