(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		milk - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(inhand milk)
	)
	(:goal (and))
)
