(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(inhand salt)
	)
	(:goal (and))
)
