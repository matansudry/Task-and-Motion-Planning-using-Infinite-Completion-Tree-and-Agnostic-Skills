(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		milk - box
		yogurt - box
		icecream - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on salt rack)
		(on icecream rack)
		(on milk table)
		(on yogurt table)
		(inhand hook)
	)
	(:goal (and))
)
