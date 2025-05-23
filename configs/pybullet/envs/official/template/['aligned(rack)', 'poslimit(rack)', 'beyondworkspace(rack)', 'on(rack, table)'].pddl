(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
	)
	(:goal (and))
)
