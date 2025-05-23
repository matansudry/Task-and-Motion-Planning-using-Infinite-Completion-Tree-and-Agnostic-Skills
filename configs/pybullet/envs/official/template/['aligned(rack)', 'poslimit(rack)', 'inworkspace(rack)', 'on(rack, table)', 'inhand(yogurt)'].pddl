(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		yogurt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(inhand yogurt)
	)
	(:goal (and))
)
