def transactionCost(nominal:float,minValue:int=1,pourcentage:float=0.0005)->float:
    return max(minValue,nominal*pourcentage)

def spreadCost(spread:float,price:float)->float:
    return price*spread/2

def sqrt_impact(orderSize:float,avgVolume:float,sigma:float,impactCoef:float=1):
    orderImpact = orderSize/avgVolume
    return impactCoef*sigma*(orderImpact**0.5)

def total_impact(nominal:float,spread:float,orderSize:float,
                 avgVolume:float,sigma:float,impactCoef:float=1,
                 minValue:int=1,pourcentage:float=0.0005)->float:
    transactioncost = transactionCost(nominal,minValue,pourcentage)
    spreadcost = spreadCost(spread,nominal)
    sqrtimpact = sqrt_impact(orderSize,avgVolume,sigma,impactCoef)
    return transactioncost+spreadcost+sqrtimpact