(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		milk - box
		yogurt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on yogurt table)
		(inhand milk)
	)
	(:goal (and))
)
