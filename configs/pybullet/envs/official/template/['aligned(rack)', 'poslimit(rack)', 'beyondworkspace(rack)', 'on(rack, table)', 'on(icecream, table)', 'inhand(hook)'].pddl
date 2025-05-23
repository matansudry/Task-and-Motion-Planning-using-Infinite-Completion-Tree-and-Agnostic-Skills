(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on icecream table)
		(inhand hook)
	)
	(:goal (and))
)
