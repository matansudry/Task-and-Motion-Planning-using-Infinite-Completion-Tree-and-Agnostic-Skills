(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(inhand icecream)
	)
	(:goal (and))
)
