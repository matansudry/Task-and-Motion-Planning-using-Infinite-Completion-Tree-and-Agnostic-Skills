(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		milk - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on milk rack)
		(inhand salt)
	)
	(:goal (and))
)
