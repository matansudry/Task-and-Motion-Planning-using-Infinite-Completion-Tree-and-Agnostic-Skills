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
		(on hook rack)
		(on milk rack)
		(on yogurt rack)
		(inhand icecream)
	)
	(:goal (and))
)
