psi_vec = c(0,0.5,1,0.25,0.75)
sig_phi = c(sqrt(2)/2,sqrt(2)*1e2/2,sqrt(2)*1e4/2,sqrt(2)*1e6/2,sqrt(2)*1e8/2,-1)
L = 333
v=0.0001
disp_vec = c(0.1,1,10)
spar_vec = c(0.1,1,10)
name.spar=c('L','M','H','X')
batch.size=10
dir = 'sELDIG_parameters/'
ticks=10000000
log=1e8
parameters = c()
sce.short = c('H','M','L')
levels_experiments = c('Low', 'Medium', 'High')
scenario = NULL
sce.short.comb.vec = NULL
for(i.letter in sce.short){
  for(j.letter in sce.short){
    sce.folder = paste0('sce',i.letter,j.letter)
    scenario = c(scenario,sce.folder)
    sce.short.comb = paste0(i.letter,j.letter)
    sce.short.comb.vec = c(sce.short.comb.vec,sce.short.comb)
  }
}
for(i in c(1,4,2,5,3)){
  for(j in c(1:6)){
    comb.temp <- paste0(i,j)
    combinations <- c(combinations,comb.temp)
  }
}

name.short <- c(1,2,3)
name.short.comb.vec <- c()
for(i.letter in name.short){
  for(j.letter in name.short){
    name.short.comb = paste0(i.letter,j.letter)
    name.short.comb.vec = c(name.short.comb.vec,name.short.comb)
  }
}

letter_count = 0
# dispersal distance
for(dis.ind in c(1:3)){
  # spatial phylogenetic distance
  for(spar.ind in c(1:3)){
    letter_count = letter_count +1
    letter.comb = name.short.comb.vec[letter_count]
    # psi
    for(i in c(1:5)){
      # phi
      for(j in c(1:6)){
        # replicates
        for(rep.sim in c(1:3)){
          simID = paste0(letter.comb,i,j,rep.sim)
          save_file = paste0(simID,'.m')
          experiments = c(levels_experiments[dis.ind],'Medium','High')
          parameters_temp = c('XE',simID, L,v,psi_vec[i],sig_phi[j],spar_vec[spar.ind],disp_vec[dis.ind],ticks,rep.sim,save_file,experiments)
          parameters = rbind(parameters,parameters_temp)
          
        }
      }
    }
    
  }
}

para_df = as.data.frame(parameters)
colnames(para_df) = c('model', 'simID', 'L', 'v', 'Psi', 's_phi', 's_spar', 's_disp', 'ticks', 'seed', 'file', 'dis','tim', 'mut')
rownames(para_df) = NULL
write.csv(para_df,paste0(dir,"XE_parameters.csv"), row.names = FALSE)
