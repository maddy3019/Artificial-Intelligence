visitedNodes = {}
    actionsToReturn = []
    fringe.push([problem.getStartState(), "Start", 0])
    while not fringe.isEmpty():
        poppedNode = fringe.pop()
        
        if len(poppedNode) > 3:
            x = poppedNode[len(poppedNode) - 1]
            node = x[0]
            action = x[1]
        else:
            node = poppedNode[0]
            action = poppedNode[1]
        
        if problem.isGoalState(node):
            start = 3
            length = len(poppedNode)
            while start < length:
                y = poppedNode[start]
                actionsToReturn.append(y[1])
                start += 1
            return actionsToReturn
                
        if node not in visitedNodes.keys():
            visitedNodes[node] = action
            successorsOfPoppedNode = problem.getSuccessors(node)
        elif node in visitedNodes.keys():
            if visitedNodes[node] != action:
                visitedNodes[node] = action
#                 successorsOfPoppedNode = problem.getSuccessors(node)
                
        for nextNode, action, cost in successorsOfPoppedNode:
            if nextNode not in visitedNodes.keys():
                copyOfPoppedNode = list(poppedNode)
                copyOfPoppedNode.append([nextNode, action, cost])
                fringe.push(copyOfPoppedNode)
            elif nextNode in visitedNodes.keys():
                if visitedNodes[nextNode]!=action:
                    copyOfPoppedNode = list(poppedNode)
                    copyOfPoppedNode.append([nextNode, action, cost])
                    fringe.push(copyOfPoppedNode)
