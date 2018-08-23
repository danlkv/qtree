class GreedyOptimiser:
    def __init__(self,num_items=1):
        self.num_items=num_items
        pass
    def optimise(self,items,func):
        result = []
        self.process = []
        for i in range(self.num_items):
            opt_cost = 999999999
            best_item = items[0]
            for item in items:
                cost = func(result+[item])
                if cost<opt_cost:
                    opt_cost = cost
                    self.process.append(cost)
                    best_item = item
            result.append(best_item)
        #return range(1,self.num_items)
        return result




