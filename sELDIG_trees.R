library(ape)
library(DDD)
dir = 'sELDIG/'
sce.short = c('H','M','L')
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

for(i_n in c(1:9)){
  scefolder = scenario[i_n]
  letter.comb = sce.short.comb.vec[i_n]
  name.comb <- name.short.comb.vec[i_n]
  for(num in combinations){
    print(paste(letter.comb, num))
    multitreefile <- paste0(dir,scefolder,'/results/1e+07/spatialpara1e+07',letter.comb,num,'/','multitree',letter.comb,num,'.tre')
    
    trees <- read.tree(multitreefile)
    for(tree_num in c(1:3)){
      save_single_tree = paste0(dir,'sELDIG_trees/XE_',name.comb,num,tree_num,'.tre')
      write.tree(trees[[tree_num]],file = save_single_tree)
    }
  }
}