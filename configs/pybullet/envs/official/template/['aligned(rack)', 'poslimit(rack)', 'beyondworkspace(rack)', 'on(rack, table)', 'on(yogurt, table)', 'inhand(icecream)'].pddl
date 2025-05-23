(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		yogurt - box
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on yogurt table)
		(inhand icecream)
	)
	(:goal (and))
)
