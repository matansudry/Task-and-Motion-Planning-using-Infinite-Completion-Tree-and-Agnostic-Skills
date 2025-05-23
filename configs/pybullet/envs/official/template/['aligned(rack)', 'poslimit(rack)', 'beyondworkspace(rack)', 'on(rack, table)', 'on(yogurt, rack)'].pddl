(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		yogurt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on yogurt rack)
	)
	(:goal (and))
)
