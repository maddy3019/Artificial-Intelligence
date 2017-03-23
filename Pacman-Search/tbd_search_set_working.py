visitedNodes = set()
    actionsToReturn = []
    fringe.push([problem.getStartState(), "Start", 0])
    while not fringe.isEmpty():
        poppedNode = fringe.pop()
        
        if len(poppedNode) > 3:
            x = poppedNode[len(poppedNode) - 1]
            node = x[0]
        else:
            node = poppedNode[0]
        
        if problem.isGoalState(node):
            start = 3
            length = len(poppedNode)
            while start < length:
                y = poppedNode[start]
                actionsToReturn.append(y[1])
                start += 1
            return actionsToReturn
                
        if node not in visitedNodes:
            visitedNodes.add(node)
            successorsOfPoppedNode = problem.getSuccessors(node)
                  
        for nextNode, action, cost in successorsOfPoppedNode:
            if nextNode not in visitedNodes:
                copyOfPoppedNode = list(poppedNode)
                copyOfPoppedNode.append([nextNode, action, cost])
                fringe.push(copyOfPoppedNode)
                
