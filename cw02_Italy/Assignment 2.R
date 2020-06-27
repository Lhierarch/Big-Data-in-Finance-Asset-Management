# Load Problem Set

WideFlows <- read.csv("Flows_Clean.csv")
WideReturns <- read.csv("Returns_Clean.csv")

# Wide - Long Panel Data Conversion

library(tidyr)
flows <- gather(WideFlows, "Stocks", "Flows", X50286:X88832)
flows <- transform(flows, Dates = as.Date(as.character(Dates), "%Y%m%d"))
returns <- gather(WideReturns, "Stocks", "Returns", X50286:X88832)
returns <- transform(returns, Dates = as.Date(as.character(Dates), "%Y%m%d"))
dt <- data.frame(merge(returns,flows))

# lag variables

library(zoo)
set.seed(123)
x1 <- zoo(dt$Returns)
x2 <- zoo(dt$Flows)
Returns_1 <- lag(x1, -1, na.pad = TRUE)
Returns_2 <- lag(x1, -2, na.pad = TRUE)
Returns_3 <- lag(x1, -3, na.pad = TRUE)
Flows_1 <- lag(x2, -1, na.pad = TRUE)
Flows_2 <- lag(x2, -2, na.pad = TRUE)
Flows_3 <- lag(x2, -3, na.pad = TRUE)
dd <- data.frame(cbind(Returns_1, Returns_2, Returns_3, Flows_1, Flows_2, Flows_3))

data <- data.frame(merge(dt, dd))
windowsSize <- 200  # training data size
testsize    <- 70    # number of observation to forecast

# load variables from the data set 
returnz      <- data$Returns  
returnz_1    <- data$Returns_1     
returnz_2    <- data$Returns_2  
returnz_3    <- data$Returns_3               
flowz_1      <- data$Flows_1
flowz_2      <- data$Flows_2
flowz_3      <- data$Flows_3
RMSE         <- matrix(0,50,1)

for(k in 0:33)  # run 34 experiments
{
  A         <- k*testsize + 1
  B         <- A + windowsSize - 1
  start_obs <- A
  end_obs   <- B
  
  returns     <- returnz[A:B]
  returns_1   <- returnz_1[A:B]
  returns_2   <- returnz_2[A:B]
  returns_3   <- returnz_3[A:B]
  flows_1     <- flowz_1[A:B]
  flows_2     <- flowz_2[A:B]
  flows_3     <- flowz_3[A:B]
  
  # ddata    <- data.frame(delta=delta, NbAlpha=NbAlpha, OSVOS=OSVOS, OSVEXT=OSVEXT)
  # output   <- paste("Gold_theta0.2per_alpha_0.1_CPTs_from", A, "to", B, 
  #                   "RollingLMTraining.csv")
  # write.csv(ddata, file=output)
  
  llmm       <- lm( returns~returns_1 + returns_2 + returns_3 + flows_1 + flows_2 + flows_3)  # initiate linear regression
  intercept  <- coef(llmm)[1]
  co_ret1   <- coef(llmm)[2]
  co_ret2 <- coef(llmm)[3]
  co_ret3   <- coef(llmm)[4]
  co_flow1   <- coef(llmm)[5]
  co_flow2   <- coef(llmm)[6]
  co_flow3   <- coef(llmm)[7]
  
  A           <- B + 1
  B           <- B + testsize
  
  returns     <- returnz[A:B]
  returns_1   <- returnz_1[A:B]
  returns_2   <- returnz_2[A:B]
  returns_3   <- returnz_3[A:B]
  flows_1     <- flowz_1[A:B]
  flows_2     <- flowz_2[A:B]
  flows_3     <- flowz_3[A:B]
  
  predict_ret <- matrix(0, testsize, 1)
  SSE         <- 0
  
  for(i in 1:testsize)  # do the forecast based on LM results
  {
    predict_ret[i] <- intercept + returns_1[i]*co_ret1 + returns_2[i]*co_ret2 + 
      returns_3[i]*co_ret3 + flows_1[i]*co_flow1 + flows_2[i]*co_flow2 + flows_3[i]*co_flow3
    SSE <- SSE + (predict_ret[i] - data$Returns[i])^2
  }
  RMSE[k+1] <- sqrt(SSE/testsize)
  # ddata    <- data.frame(Predicted_ret=predict_ret, Real_ret=data$Returns)
  # output   <- paste("from", A, "to", B, "RollingLMTesting.csv")
  # write.csv(ddata, file=output)
  print(RMSE[k+1])
  
}

