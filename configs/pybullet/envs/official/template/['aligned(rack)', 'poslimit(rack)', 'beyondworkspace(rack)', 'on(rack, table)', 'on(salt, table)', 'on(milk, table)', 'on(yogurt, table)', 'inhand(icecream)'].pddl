(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
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
		(on salt table)
		(on milk table)
		(on yogurt table)
		(inhand icecream)
	)
	(:goal (and))
)
