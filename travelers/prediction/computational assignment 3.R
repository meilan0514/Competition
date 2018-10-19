# HI - Please read through this code and try to understand each line of code.
# An simulation assignment is at the end. 
# Coming attraction: 
# We'll have some "real data" analysis next week! 

# clear memory
rm(list=ls())

# this library has mixed model estimation routines
library("nlme")

# Set parameters
n <- 500  # number of data points
sig <- 1.1 # std dev of regression error
num.knots <- min(30,floor(n/3)) # number of knots in spline

# Generate data (i.e. scatterplot of x1 versus y)
x1 <- runif(n)
eps <- rnorm(n)
f1 <- function(x) return(1.5*dnorm(x,.3,.1)-dnorm(x,.75,0.07))
y <- f1(x1) + sig*eps

# Set up design matrices and random effects block structure
knots.1 <- seq(min(x1),max(x1),length=(num.knots+2))[-c(1,num.knots+2)]

# use a quadratic spline
X <- cbind(rep(1,n),x1,x1^2)
Z.1 <- outer(x1,knots.1,"-")
Z.1 <- Z.1*(Z.1>0)
Z <- cbind(Z.1^2)

C.mat <- cbind(X,Z)
re.block.val <- list(1:num.knots)

Z.block  <-  list()
for (i in 1:length(re.block.val))
  Z.block[[i]] <-
  as.formula(paste("~Z[,c(",
                   paste(re.block.val[[i]],collapse=","),")]-1"))

# Fit model using lme() and extract coefficient estimates

data.fr <- groupedData( y ~ X[,-1] | rep(1,length=length(y)),
                       data = data.frame( y,X,Z))

lme.fit <-
  lme(y~X[,-1],
      data=data.fr,
      random=pdMat(Z.block[[1]],pdClass="pdIdent"))

u.hat <- as.vector(unlist(lme.fit$coef$ran))
beta.hat <- as.vector(unlist(lme.fit$coef$fix))

sigusq.hat <-
  (as.numeric(exp(attributes(summary(lme.fit)$apVar)$Pars[1])))^2
sigesq.hat <- (lme.fit$sigma^2)

# Draw fits 
grid.size <- 101
x1.grid <- seq(0,1,length=grid.size)
ones.grid <- rep(1,grid.size)

X.grid <- cbind(ones.grid,x1.grid,x1.grid^2)
Z.grid <- outer(x1.grid,knots.1,"-")
Z.grid <- Z.grid*(Z.grid>0)
Z.grid <- Z.grid^2

f1.hat.grid <- as.vector(X.grid%*%beta.hat+ Z.grid%*%u.hat)
f1.grid <- f1(x1.grid)

par(bty="l")

plot(c(x1.grid),c(f1.hat.grid),bty="l",xlab="x1",
  ylab="f(x)",type="n",ylim=range(c(y,f1.grid,f1.hat.grid)))

points(x1,y,pch=16) # pch is type of point
lines(x1.grid,f1.grid) 
lines(x1.grid,f1.hat.grid,lwd=2,col=2)



# Do a Monte Carlo simulation to compare mixed model
# method of estimating a spline with one that uses 
# cross validation to estimate the smoothing parameter.
# Compare the methods by rMISE, squared bias, and variance.
