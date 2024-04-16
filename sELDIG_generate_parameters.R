
# Example dataframe of parameters and min/max values

paramLimits = data.frame(parameter = c('psi', 'sig_phi', 'dispersal', 'spatial'),
                         min = c(0, 0, log(0.1), log(0.1)),
                         max = c(1, 1, log(100), log(100)))


# Function for sampling parameter space

sampleParams = function(paramLimits, n = 1000, seed = 1, startingSimID = 1) {
  set.seed(seed)
  
  output = data.frame(simID = startingSimID:(startingSimID + n - 1))
  
  for (i in 1:nrow(paramLimits)) {
    output = cbind(output, 
                   data.frame(runif(n)*(paramLimits$max[i] - paramLimits$min[i]) + paramLimits$min[i]))
  }
  names(output)[2:ncol(output)] = as.character(paramLimits$parameter)
  
  return(output)
}

parameters <- sampleParams(paramLimits)
parameters$dispersal <- exp(parameters$dispersal)
parameters$spatial <- exp(parameters$spatial)

psi_vec = parameters$psi
sig_phi = parameters$sig_phi
L = 333
v=0.0001
disp_vec = parameters$dispersal
spar_vec = parameters$spatial

batch.size=10
dir = "sELDIG/parameters"
ticks=10000000
log=1e8


dir.scefolder = paste0(dir,"/sce")
dir.folder = paste0(dir.scefolder,"/results")
dir.create(file.path(dir.scefolder), showWarnings = FALSE)
dir.create(file.path(dir.folder), showWarnings = FALSE)

append = TRUE

setwd(file.path(dir.folder))
count = 0
for(rep.sim in c(1:nrow(parameters))){
  
  str = sprintf('L=%i v=%.4f Psi=%.4f s_phi=%.4f s_spar=%.1f s_disp=%.1f ticks=%i seed=%i file=results/xe_%i.m',
                L,v,psi_vec[rep.sim],sig_phi[rep.sim],spar_vec[rep.sim],disp_vec[rep.sim],ticks,1,rep.sim)
  if(count%%batch.size == 0){
    no.batch = count%/%batch.size+1
    filename<-paste0(dir.scefolder,"/spatialpara",formatC(ticks),"batch",no.batch,".txt")
    
    # write(str, file=filename,append=FALSE)
    cat(str,'\n', file=filename, append=FALSE, sep='')
  }else{
    cat(str ,'\n', file=filename, append=append, sep='')
    print(str)
  }
  count = count +1
  
}


