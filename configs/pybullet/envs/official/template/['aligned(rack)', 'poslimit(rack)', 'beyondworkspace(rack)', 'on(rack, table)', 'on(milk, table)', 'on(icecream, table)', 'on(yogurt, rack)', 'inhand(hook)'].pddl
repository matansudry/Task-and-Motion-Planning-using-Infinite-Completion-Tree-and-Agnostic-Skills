(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		milk - box
		yogurt - box
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on milk table)
		(on icecream table)
		(on yogurt rack)
		(inhand hook)
	)
	(:goal (and))
)
