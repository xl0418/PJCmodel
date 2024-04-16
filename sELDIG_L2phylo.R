library(ape)
library(DDD)
dir = 'sELDIG/'
ev2l <- function(ltable, cond){
  ancestor0 <- cond[1:2]
  ancestor1 <- cond[3:4]
  
  row1 <- c(10^7, 0.0, 1.0, ancestor0[2])
  row2 <- c(10^7, 1, 2, ancestor1[2])
  ltable[,2] <- ltable[,2] + 1
  ltable[,3] <- ltable[,3] + 1
  ltable <- rbind(row1, row2, ltable)
  phylo_L <- DDD::L2phylo(ltable, dropextinct = TRUE)
  return(phylo_L)
}



for(i_n in c(1:1000)){
  rname = paste0(dir,'results/xe_', i_n, '.csv')
  L.table = read.csv(rname,header = FALSE)
  L.table <- as.matrix(L.table)
  ancestor_con <- L.table[1,]
  L.table <- L.table[2:nrow(L.table),]
  # print(nrow(L.table))
  single.phylo = ev2l(L.table, ancestor_con)

  singletreefile <- paste0(dir, 'trees/xe_',i_n,'.tre')
  
  write.tree(single.phylo,file=singletreefile)

}